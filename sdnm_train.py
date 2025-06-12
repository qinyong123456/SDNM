import copy
import torch
import torchvision
from sklearn.cluster import KMeans
from torch import nn
from lightly.loss.memory_bank import MemoryBankModule
from lightly.models import utils
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
from lightly.transforms.smog_transform import SMoGTransform
from lightly.loss import NTXentLoss
from typing import Dict, Tuple
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.optim import SGD
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor, LearningRateMonitor
import numpy as np
import os
import shutil
import random

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(42)

# 检查并安装Lightly库
try:
    import lightly
except ImportError:
    !pip install lightly

# 检查并安装Kaggle API
try:
    import kaggle
except ImportError:
    !pip install kaggle

# 自动下载EuroSAT数据集
def download_eurosat_dataset():
    """自动下载EuroSAT数据集到Kaggle环境"""
    dataset_path = '/kaggle/input/eurosat-dataset'
    
    # 检查数据集是否已存在
    if os.path.exists(dataset_path):
        print(f"数据集已存在于 {dataset_path}")
        return dataset_path
    
    print("正在下载EuroSAT数据集...")
    
    # 创建输入目录
    os.makedirs('/kaggle/input', exist_ok=True)
    
    # 下载数据集
    try:
        # 方法1: 使用Kaggle API命令行工具
        !kaggle datasets download -d ameroyer/eurosat-dataset -p /kaggle/input/
        
        # 解压数据集
        import zipfile
        with zipfile.ZipFile('/kaggle/input/eurosat-dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('/kaggle/input/eurosat-dataset')
        
        print("数据集下载和解压完成!")
        return '/kaggle/input/eurosat-dataset'
    except Exception as e:
        print(f"下载数据集时出错: {e}")
        print("尝试从Kaggle Datasets直接下载...")
        
        # 方法2: 使用Kaggle Python库
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()  # 通常在Kaggle环境中已自动认证
            api.dataset_download_files('ameroyer/eurosat-dataset', path='/kaggle/input', unzip=True)
            print("数据集下载和解压完成!")
            return '/kaggle/input/eurosat-dataset'
        except Exception as e2:
            print(f"再次尝试下载数据集时出错: {e2}")
            print("请确保您已在Kaggle设置中启用了Internet访问，并接受了数据集的使用条款")
            raise

# 检查是否有GPU可用
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
print(f"Using accelerator: {accelerator}")
devices_num = 1
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

# 定义参数类
class Args:
    nThreads = 4
    batch_size = 256
    input_size = 256
    feature_dim = 512
    # 数据集路径将由download_eurosat_dataset函数设置
    path_to_data = None
    eval_max_epochs = 50
    # 添加输出路径
    output_dir = '/kaggle/working/'
    # 设置随机种子
    seed = 42

args = Args()

# 下载数据集
args.path_to_data = download_eurosat_dataset()

# 定义SMoG模型
class SMoGModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SMoGProjectionHead(512, 2048, 128)
        self.prediction_head = SMoGPredictionHead(128, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)
        self.n_groups = 300
        self.smog = SMoGPrototypes(
            group_features=torch.rand(self.n_groups, 128), beta=0.99
        )

    def _cluster_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.cpu().numpy()
        kmeans = KMeans(self.n_groups).fit(features)
        clustered = torch.from_numpy(kmeans.cluster_centers_).float()
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        return clustered

    def reset_group_features(self, memory_bank):
        features = memory_bank.bank
        group_features = self._cluster_features(features.t())
        self.smog.set_group_features(group_features)

    def reset_momentum_weights(self):
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        encoded = self.projection_head(features)
        predicted = self.prediction_head(encoded)
        return encoded, predicted

    def forward_momentum(self, x):
        features = self.backbone_momentum(x).flatten(start_dim=1)
        encoded = self.projection_head_momentum(features)
        return encoded

# 初始化模型
resnet = torchvision.models.resnet18()
backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
model = SMoGModel(backbone)
memory_bank_size = 300 * args.batch_size
memory_bank = MemoryBankModule(size=memory_bank_size)

# 移动模型到GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Moving model to {device}")
model.to(device)

# 数据加载和预处理
image_mean, image_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.input_size),
    torchvision.transforms.CenterCrop(args.input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(image_mean, image_std)
])

# 检查数据路径是否存在
if not os.path.exists(args.path_to_data):
    raise FileNotFoundError(f"找不到数据路径: {args.path_to_data}")

print(f"数据集路径: {args.path_to_data}")
print("数据集中的内容:")
!ls {args.path_to_data}

# 加载数据集
rs_data = torchvision.datasets.ImageFolder(args.path_to_data, transform)
num_classes = len(rs_data.classes)
print(f"数据集包含 {num_classes} 个类别")

# 设置随机种子以确保数据集划分与原始代码一致
torch.manual_seed(args.seed)

len_rs_data = len(rs_data)
num_test = int(0.2 * len_rs_data)
num_train = int(0.6 * len_rs_data)

# 使用相同的随机种子进行数据集划分
train_rs_data, test_rs_data = torch.utils.data.random_split(
    rs_data, 
    lengths=[num_train, num_test],
    generator=torch.Generator().manual_seed(args.seed)
)

print(f"训练集大小: {len(train_rs_data)}")
print(f"测试集大小: {len(test_rs_data)}")

tr_data_loader = torch.utils.data.DataLoader(
    train_rs_data,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.nThreads
)

# 定义损失函数和优化器
global_criterion = nn.CrossEntropyLoss()
local_criterion = NTXentLoss()

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6
)

# 训练循环
global_step = 0
n_epochs = 30
noise_factor = 0.5
model.train()

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)

print("Starting Training")
for epoch in range(n_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(tr_data_loader):
        x0 = batch[0]
        noisy_imgs = x0 + noise_factor * torch.randn(*x0.shape)
        x1 = np.clip(noisy_imgs, 0., 1.)

        if batch_idx % 2:
            x1, x0 = x0, x1

        x0 = x0.to(device)
        x1 = x1.to(device)

        if global_step > 0 and global_step % 300 == 0:
            model.reset_group_features(memory_bank=memory_bank)
            model.reset_momentum_weights()
        else:
            utils.update_momentum(model.backbone, model.backbone_momentum, 0.99)
            utils.update_momentum(model.projection_head, model.projection_head_momentum, 0.99)

        x0_encoded, x0_predicted = model(x0)
        x1_encoded = model.forward_momentum(x1)

        assignments = model.smog.assign_groups(x1_encoded)
        group_features = model.smog.get_updated_group_features(x0_encoded)
        logits = model.smog(x0_predicted, group_features, temperature=0.1)
        model.smog.set_group_features(group_features)

        loss = 0.5 * global_criterion(logits, assignments) + 0.5 * local_criterion(x0_encoded, x1_encoded)

        memory_bank(x0_encoded, update=True)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        total_loss += loss.detach()

    avg_loss = total_loss / len(tr_data_loader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    
    # 每个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        model_path = f"{args.output_dir}/smog_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")

# 保存最终模型
final_model_path = f"{args.output_dir}/smog_model_final.pth"
torch.save(model.state_dict(), final_model_path)
print(f"最终模型已保存到: {final_model_path}")

# 定义MLP分类器
class MLPClassifier(LightningModule):
    def __init__(
        self,
        model: Module,
        batch_size: int,
        feature_dim: int = 2048,
        num_classes: int = 1000,
        topk: Tuple[int, ...] = (1, 5),
        freeze_model: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.topk = topk
        self.freeze_model = freeze_model

        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, num_classes),
        )

        self.criterion = CrossEntropyLoss()

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images).flatten(start_dim=1)
        return self.classification_head(features)

    def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
        images, targets = batch[0], batch[1]
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log(
            "train_loss", loss, batch_size=batch_size
        )
        self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        parameters = list(self.classification_head.parameters())
        if not self.freeze_model:
            parameters += self.backbone.parameters()
        optimizer = SGD(
            parameters,
            lr=0.1 * self.batch_size * self.trainer.world_size / 256,
            momentum=0.9,
            weight_decay=0.0,
        )
        scheduler = {
            "scheduler": CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_epochs=0,
                max_epochs=self.trainer.estimated_stepping_batches,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

# 定义评估函数
def MLP_eval(model, batch_size, num_workers, accelerator, devices, num_classes, linear_tr_loader, linear_te_loader, ft_max_epochs):
    metric_callback = MetricCallback()
    trainer = pl.Trainer(
        max_epochs=ft_max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
    )
    classifier = MLPClassifier(
        model=model,
        batch_size=batch_size,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        freeze_model=False,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=linear_tr_loader,
        val_dataloaders=linear_te_loader,
    )

    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(
            f"max classification {metric}: {max(metric_callback.val_metrics[metric])}"
        )

    return trainer.logged_metrics

# 定义创建评估数据加载器的函数
def create_data_loader_eval(input_size, batch_size, num_workers):
    _transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(image_mean, image_std)
    ])

    # 重新加载数据集并使用相同的随机种子进行划分
    rs_data_set = torchvision.datasets.ImageFolder(args.path_to_data, transform=_transform)
    
    # 确保使用相同的随机种子
    torch.manual_seed(args.seed)
    
    len_rs_data = len(rs_data_set)
    num_test = int(0.2 * len_rs_data)
    num_train = int(0.6 * len_rs_data)

    train_rs_data, test_rs_data = torch.utils.data.random_split(
        rs_data_set, 
        lengths=[num_train, num_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    rs_train_loader = torch.utils.data.DataLoader(
        train_rs_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    rs_test_loader = torch.utils.data.DataLoader(
        test_rs_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return rs_train_loader, rs_test_loader

# 创建评估数据加载器
rs_train_loader, rs_test_loader = create_data_loader_eval(args.input_size, args.batch_size, args.nThreads)

# 评估模型
print("Starting model evaluation...")
model.eval()
logged_metrics = MLP_eval(model, args.batch_size, args.nThreads, accelerator, devices_num, num_classes, rs_train_loader,
                          rs_test_loader, args.eval_max_epochs)
print(logged_metrics)

# 保存评估结果
import json
results_path = f"{args.output_dir}/evaluation_results.json"
with open(results_path, 'w') as f:
    json.dump({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in logged_metrics.items()}, f)
print(f"评估结果已保存到: {results_path}")
