"""
PyTorch Training Loop - Complete Training Pipeline
===================================================

This module covers the complete training workflow including data loading,
optimization, loss functions, and training/evaluation loops.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===========================================================================
# LOSS FUNCTIONS
# ===========================================================================

# Classification losses
cross_entropy = (
    nn.CrossEntropyLoss()
)  # multi-class (expects logits, not softmax)
nll_loss = nn.NLLLoss()  # expects log probabilities
bce_loss = nn.BCELoss()  # binary cross entropy (expects sigmoid output)
bce_with_logits = nn.BCEWithLogitsLoss()  # binary CE with built-in sigmoid

# Regression losses
mse_loss = nn.MSELoss()  # mean squared error (L2)
l1_loss = nn.L1Loss()  # mean absolute error (L1)
smooth_l1 = nn.SmoothL1Loss()  # Huber loss
huber = nn.HuberLoss(delta=1.0)  # same as SmoothL1

# Other losses
kl_div = nn.KLDivLoss(reduction="batchmean")  # KL divergence
cosine_loss = nn.CosineEmbeddingLoss()  # cosine similarity based

# Loss with reduction options
# 'none': no reduction (returns element-wise loss)
# 'mean': mean of all elements (default)
# 'sum': sum of all elements
loss_fn = nn.CrossEntropyLoss(reduction="mean")

# Label smoothing for better generalization
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# Class weights for imbalanced datasets
weights = torch.tensor([1.0, 2.0, 0.5])  # weight for each class
weighted_ce = nn.CrossEntropyLoss(weight=weights)

# ===========================================================================
# OPTIMIZERS
# ===========================================================================

model = nn.Linear(10, 2)

# Stochastic Gradient Descent
sgd = optim.SGD(model.parameters(), lr=0.01)
sgd_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
sgd_nesterov = optim.SGD(
    model.parameters(), lr=0.01, momentum=0.9, nesterov=True
)
sgd_wd = optim.SGD(
    model.parameters(), lr=0.01, weight_decay=1e-4
)  # L2 regularization

# Adam and variants
adam = optim.Adam(model.parameters(), lr=0.001)
adam_wd = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
adamw = optim.AdamW(
    model.parameters(), lr=0.001, weight_decay=0.01
)  # decoupled WD

# Other optimizers
rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
adagrad = optim.Adagrad(model.parameters(), lr=0.01)

# Per-parameter options (different LR for different layers)
optimizer = optim.Adam(
    [
        {"params": model.weight, "lr": 1e-3},
        {"params": model.bias, "lr": 1e-4},
    ]
)

# ===========================================================================
# LEARNING RATE SCHEDULERS
# ===========================================================================

model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step decay: multiply LR by gamma every step_size epochs
step_scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=30, gamma=0.1
)

# Multi-step decay: multiply by gamma at specific epochs
multistep = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 80], gamma=0.1
)

# Exponential decay
exponential = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing (smooth decay to min_lr)
cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=100, eta_min=1e-6
)

# Cosine with warm restarts
cosine_warm = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)

# Reduce on plateau (reduce when metric stops improving)
plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.1, patience=10
)


# Linear warmup + decay (common in transformers)
def lr_lambda(current_step):
    warmup_steps = 1000
    if current_step < warmup_steps:
        return current_step / warmup_steps
    return max(0.0, 1.0 - (current_step - warmup_steps) / 10000)


warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# One cycle policy (fast.ai style)
onecycle = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=100, epochs=10
)

# ===========================================================================
# CUSTOM DATASET
# ===========================================================================


class CustomDataset(Dataset):
    """Example custom dataset."""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Create synthetic dataset
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))

dataset = CustomDataset(X, y)

# Or use TensorDataset for simple cases
tensor_dataset = TensorDataset(X, y)

# ===========================================================================
# DATA LOADING
# ===========================================================================

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,  # shuffle training data
    num_workers=0,  # parallel data loading (0 for main process)
    pin_memory=True,  # faster GPU transfer
    drop_last=True,  # drop incomplete last batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,  # can use larger batch for validation
    shuffle=False,  # don't shuffle validation
    num_workers=0,
    pin_memory=True,
)

# ===========================================================================
# BASIC TRAINING LOOP
# ===========================================================================


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()  # clear gradients
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient computation
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


# ===========================================================================
# COMPLETE TRAINING SCRIPT
# ===========================================================================


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_model(num_epochs=20):
    """Complete training pipeline."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SimpleClassifier(input_dim=20, num_classes=3).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    # Best model tracking
    best_val_loss = float("inf")
    best_model_state = None

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Print progress
        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation loss: {best_val_loss:.4f}")

    return model, history


# ===========================================================================
# GRADIENT CLIPPING (for RNNs and unstable training)
# ===========================================================================


def train_with_gradient_clipping(
    model, loader, criterion, optimizer, device, max_norm=1.0
):
    """Training with gradient clipping."""
    model.train()

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Clip gradients by norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # Or clip by value
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

        optimizer.step()


# ===========================================================================
# MIXED PRECISION TRAINING (for faster GPU training)
# ===========================================================================

from torch.amp import autocast, GradScaler


def train_mixed_precision(model, loader, criterion, optimizer, device):
    """Training with automatic mixed precision."""
    model.train()
    scaler = GradScaler("cuda")

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast("cuda"):
            output = model(data)
            loss = criterion(output, target)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


# ===========================================================================
# GRADIENT ACCUMULATION (for larger effective batch size)
# ===========================================================================


def train_with_accumulation(
    model, loader, criterion, optimizer, device, accumulation_steps=4
):
    """Gradient accumulation for larger effective batch size."""
    model.train()
    optimizer.zero_grad()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps  # normalize loss
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()


# ===========================================================================
# EARLY STOPPING
# ===========================================================================


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        return False


# Usage:
# early_stopping = EarlyStopping(patience=10)
# for epoch in range(epochs):
#     train_loss = train_epoch(...)
#     val_loss = validate(...)
#     if early_stopping(val_loss, model):
#         print("Early stopping triggered")
#         break

# ===========================================================================
# CHECKPOINTING
# ===========================================================================


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path):
    """Save training checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler else None
            ),
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"], checkpoint["loss"]


if __name__ == "__main__":
    print("Starting training demo...")
    model, history = train_model(num_epochs=10)
    print("\nTraining complete!")
