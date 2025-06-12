import copy
import torch
import torchvision
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

from lightly.loss.memory_bank import MemoryBankModule
from lightly.models import utils
from lightly.models.modules.heads import (
    SMoGPredictionHead,
    SMoGProjectionHead,
    SMoGPrototypes,
)
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

# 设置Kaggle环境
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices_num = 1 if accelerator == "gpu" else "auto"

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

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

# 参数设置
class Args:
    nThreads = 4
    batch_size = 256
    input_size = 64  # EuroSAT图像大小为64x64
    feature_dim = 512
    path_to_data = '/kaggle/input/eurosat-dataset/EuroSAT'  # Kaggle上的路径
    eval_max_epochs = 50

args = Args()

# 加载EuroSAT数据集
def load_eurosat_dataset():
    # 图像归一化参数
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    # 加载数据集
    full_dataset = datasets.ImageFolder(args.path_to_data, transform=transform)
    num_classes = len(full_dataset.classes)
    
    # 划分数据集
    dataset_size = len(full_dataset)
    train_size = int(0.6 * dataset_size)
    test_size = int(0.2 * dataset_size)
    val_size = dataset_size - train_size - test_size
    
    train_dataset, test_dataset, val_dataset = random_split(
        full_dataset, [train_size, test_size, val_size]
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}, Val: {len(val_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    return train_dataset, test_dataset, val_dataset, num_classes

# 加载数据
train_dataset, test_dataset, val_dataset, num_classes = load_eurosat_dataset()

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.nThreads,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.nThreads,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.nThreads,
    pin_memory=True
)

# 初始化模型
resnet = torchvision.models.resnet18()
backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
model = SMoGModel(backbone)
memory_bank_size = 300 * args.batch_size
memory_bank = MemoryBankModule(size=memory_bank_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 训练参数
global_criterion = nn.CrossEntropyLoss()
local_criterion = NTXentLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6
)
global_step = 0
n_epochs = 30
noise_factor = 0.5

# 训练循环
print("Starting Training")
model.train()
for epoch in range(n_epochs):
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x0, _ = batch  # 忽略标签
        noisy_imgs = x0 + noise_factor * torch.randn(*x0.shape)
        x1 = torch.clamp(noisy_imgs, 0., 1.)  # 使用torch.clamp替代np.clip

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

    avg_loss = total_loss / len(train_loader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

# 分类器部分保持不变
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
        images, targets = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, targets)
        _, predicted_labels = predictions.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
        return loss, topk

    def training_step(self, batch, batch_idx) -> Tensor:
        loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
        batch_size = len(batch[1])
        log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
        self.log("train_loss", loss, batch_size=batch_size)
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

# 评估函数
def evaluate_model(model, train_loader, val_loader):
    metric_callback = MetricCallback()
    trainer = pl.Trainer(
        max_epochs=args.eval_max_epochs,
        accelerator=accelerator,
        devices=devices_num,
        callbacks=[
            LearningRateMonitor(),
            DeviceStatsMonitor(),
            metric_callback,
        ],
    )
    
    classifier = MLPClassifier(
        model=model,
        batch_size=args.batch_size,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        freeze_model=False,
    )
    
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(
            f"max classification {metric}: {max(metric_callback.val_metrics[metric])}"
        )
    
    return trainer.logged_metrics

# 运行评估
model.eval()
logged_metrics = evaluate_model(model, train_loader, val_loader)
print(logged_metrics)
