"""
PointNet for ModelNet40 Classification
Based on: "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
arxiv: 1612.00593, Appendix C

Training recipe exactly as described in the paper:
- 1024 points uniformly sampled, normalized to unit sphere
- Data augmentation: random rotation around up-axis + jitter (σ=0.02)
- Adam lr=0.001, batch size 32, lr divided by 2 every 20 epochs
- Weight decay for BN: starts at 0.5, increases to 0.99
- Dropout keep ratio 0.7 on last FC (256)
- Orthogonal regularization weight 0.001 on T-Net matrices
"""

import os
import math
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

# Le librerie per scaricare dati e registrare il training sono importate in main:
# per usare i pesi gia addestrati bastano NumPy e PyTorch.
from torch.utils.data import DataLoader, Dataset

# ============================================================
# PointNet Architecture
# ============================================================

class TNet(nn.Module):
    # Impara una matrice: 3x3 sulle coordinate, oppure 64x64 sulle feature.
    # Non e una camera e non ricostruisce una superficie 3D.
    """Transformation Network (mini-PointNet predicting a k×k matrix)."""
    # Scopo: Costruisce una mini-PointNet che inizialmente restituisce la matrice identità.
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        # Initialize output as identity matrix
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.copy_(torch.eye(k).flatten())

    # Scopo: Riassume i punti e produce una matrice k×k distinta per ogni oggetto del batch.
    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False)[0]  # global max pool
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x.view(bs, self.k, self.k)


class PointNetClassification(nn.Module):
    """PointNet for 3D object classification (ModelNet40)."""
    # Scopo: Definisce trasformazioni, MLP condivisi e classificatore per l’intero oggetto.
    def __init__(self, num_classes=40, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        # Input transform (3x3)
        self.input_transform = TNet(k=3)

        # Shared MLP after input transform
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

        # Feature transform (64x64)
        self.feature_transform = TNet(k=64)

        # Shared MLP after feature transform
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

    # Scopo: Trasforma coordinate e feature, aggrega i punti e restituisce logits e matrici.
    def forward(self, x):
        # x: (B, 3, N) point cloud
        # B = numero di oggetti; 3 = coordinate XYZ; N = punti per oggetto.
        bs = x.size(0)

        # Input transform
        trans_3x3 = self.input_transform(x)
        x = torch.bmm(trans_3x3, x)  # apply transform

        # Shared MLP (64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        trans_64x64 = self.feature_transform(x)
        x = torch.bmm(trans_64x64, x)

        # Shared MLP (64, 128, 1024)
        # Conv1d con kernel 1 applica gli stessi pesi a ogni punto separatamente.
        # Ogni punto diventa un vettore di 1024 caratteristiche apprese.
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        # Global max pooling → (B, 1024)
        # Il massimo per canale riassume tutti i punti in 1024 valori globali.
        # L'ordine dei punti non influenza questo aggregatore.
        x = torch.max(x, dim=2, keepdim=False)[0]

        # Classifier
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)
        # 40 punteggi grezzi (logits). Softmax serve solo per presentarli come
        # probabilita relative; non e una misura certa della correttezza.
        return x, trans_3x3, trans_64x64


# ============================================================
# Data Loading & Augmentation
# ============================================================

# Aumenta la variabilita del training con rotazioni e rumore; non usata nella nostra inferenza.
# Scopo: Nel training originale aumenta i dati con rotazioni attorno a Z e rumore gaussiano.
def augment_pointcloud(pc, train=True):
    """Apply augmentations as described in Section 5.1 of the PointNet paper."""
    if not train:
        return pc
    batch_size, num_points, _ = pc.shape
    # 1. Random rotation around up-axis (z-axis)
    theta = torch.rand(batch_size, device=pc.device) * 2 * math.pi
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros(batch_size, device=pc.device)
    ones = torch.ones(batch_size, device=pc.device)
    rot = torch.stack([cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones], dim=1)
    rot = rot.view(batch_size, 3, 3)
    pc = torch.bmm(pc, rot.transpose(1, 2))  # rotate each point
    # 2. Jitter with Gaussian noise (σ=0.02)
    jitter = torch.randn_like(pc) * 0.02
    pc = pc + jitter
    return pc


class ModelNet40Dataset(Dataset):
    """Wrap HuggingFace ModelNet40 dataset."""
    # Scopo: Configura il dataset originale e il numero di punti richiesto dal modello.
    def __init__(self, dataset, num_points=1024, train=True):
        self.data = dataset
        self.num_points = num_points
        self.train = train

    # Scopo: Comunica al DataLoader quanti oggetti sono disponibili nello split.
    def __len__(self):
        return len(self.data)

    # Scopo: Prepara un oggetto del training originale con campionamento e normalizzazione.
    def __getitem__(self, idx):
        sample = self.data[idx]
        points = np.array(sample['inputs'], dtype=np.float32)  # shape: (2048, 3) or (N, 3)

        # Subsample to num_points
        n = points.shape[0]
        if n >= self.num_points:
            indices = np.random.choice(n, self.num_points, replace=False)
        else:
            indices = np.random.choice(n, self.num_points, replace=True)
        points = points[indices]

        # Center and normalize to unit sphere (as paper: normalize into unit sphere)
        centroid = points.mean(axis=0)
        points = points - centroid
        max_norm = np.linalg.norm(points, axis=1).max()
        if max_norm > 0:
            points = points / max_norm

        label = sample['label']

        # Convert to (3, N) format for PointNet
        points = torch.from_numpy(points).float().transpose(0, 1)  # (3, N)
        label = torch.tensor(label, dtype=torch.long)
        return points, label


# ============================================================
# Training
# ============================================================

# Penalizza matrici lontane dall ortogonalita; non impone una rotazione esatta.
# Scopo: Penalizza la distanza di M Mᵀ dall’identità; non impone una rotazione esatta.
def orthogonality_loss(mat):
    """Regularization loss to keep transformation matrix close to orthogonal."""
    bs = mat.size(0)
    k = mat.size(1)
    identity = torch.eye(k, device=mat.device).unsqueeze(0).expand(bs, k, k)
    return torch.mean(torch.norm(torch.bmm(mat, mat.transpose(1, 2)) - identity, dim=(1, 2)))


# Training originale completo: forward, loss, backpropagation e aggiornamento dei pesi.
# Scopo: Esegue un’epoca di training completo: augmentation, loss, gradienti e aggiornamento.
def train_epoch(model, loader, optimizer, device, orthogonal_weight=0.001):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        bs = points.size(0)

        # Augmentation (rotate + jitter)
        points = augment_pointcloud(points.transpose(1, 2).contiguous(), train=True)
        points = points.transpose(1, 2).contiguous()  # back to (B, 3, N)

        optimizer.zero_grad()

        logits, trans_3x3, trans_64x64 = model(points)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Orthogonal regularization on both transforms
        ortho_loss = orthogonality_loss(trans_3x3) + orthogonality_loss(trans_64x64)
        loss = cls_loss + orthogonal_weight * ortho_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        total_acc += (pred == labels).sum().item()
        total += bs

    return total_loss / total, total_acc / total


# Valutazione senza gradienti: misura loss e accuracy senza apprendere dai dati di test.
# Scopo: Misura loss e accuracy con rete in eval, senza aggiornamenti dei pesi.
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        bs = points.size(0)

        logits, _, _ = model(points)
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * bs
        pred = logits.argmax(dim=1)
        total_acc += (pred == labels).sum().item()
        total += bs

    return total_loss / total, total_acc / total


# ============================================================
# Main
# ============================================================

# Entry point del training originale. L app importa le classi senza eseguire questa funzione.
# Scopo: Avvia il training completo fornito dall’autore; l’app locale non lo esegue.
def main():
    # Questa funzione avvia il training originale; l'app non la chiama.
    import trackio
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--orthogonal_weight', type=float, default=0.001)
    parser.add_argument('--lr_decay_epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dataset', type=str, default='jxie/modelnet40-2048')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--push_to_hub', action='store_true')
    parser.add_argument('--hub_model_id', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize trackio
    trackio.init(
        project=os.environ.get("TRACKIO_PROJECT", "pointnet-modelnet40"),
        name=f"pointnet_lr{args.lr}_bs{args.batch_size}_pts{args.num_points}",
        config=vars(args),
    )

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    train_ds = ModelNet40Dataset(ds['train'], num_points=args.num_points, train=True)
    test_ds = ModelNet40Dataset(ds['test'], num_points=args.num_points, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    # Model
    model = PointNetClassification(num_classes=40, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer: Adam as per paper
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.999))  # "momentum 0.9" → β1=0.9

    # LR scheduler: divide by 2 every 20 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=0.5)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device,
                                             orthogonal_weight=args.orthogonal_weight)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

        trackio.log({
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'test/loss': test_loss,
            'test/accuracy': test_acc,
            'lr': current_lr,
        }, step=epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'args': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ New best model (acc: {test_acc*100:.2f}%)")

    print(f"\nTraining complete. Best test accuracy: {best_acc*100:.2f}%")
    trackio.log({'best/test_accuracy': best_acc}, step=args.epochs)
    trackio.finish()

    # Save final model in HF format
    if args.push_to_hub:
        from huggingface_hub import HfApi
        hub_id = args.hub_model_id or "DavidHanSZ/pointnet-modelnet40"
        api = HfApi()
        os.makedirs(args.output_dir, exist_ok=True)

        # Save model with config
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'pytorch_model.bin'))

        config = {
            'architectures': ['PointNetClassification'],
            'num_classes': 40,
            'num_points': args.num_points,
            'dropout': args.dropout,
        }
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        api.upload_file(
            path_or_fileobj=os.path.join(args.output_dir, 'pytorch_model.bin'),
            path_in_repo='pytorch_model.bin',
            repo_id=hub_id,
            repo_type='model',
        )
        api.upload_file(
            path_or_fileobj=os.path.join(args.output_dir, 'config.json'),
            path_in_repo='config.json',
            repo_id=hub_id,
            repo_type='model',
        )
        print(f"Model pushed to: https://huggingface.co/{hub_id}")


if __name__ == '__main__':
    main()
