# Gemini_v3.py
# Based on Claude_v2.py, incorporating further improvements and experimental setups

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import random
from torch.optim.lr_scheduler import _LRScheduler

# =============================================================================
# Configuration
# =============================================================================
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# 1. Data Preprocessing and Augmentation
# =============================================================================

def get_transforms(config):
    """Returns data transformation pipelines based on the config."""
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR10 statistics

    if not config.get('use_augmentation', False):
        # Basic transform - normalization only for testing or baseline
        return transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # Enhanced data augmentation transforms
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if config.get('use_rotation', False):
        transform_list.append(transforms.RandomRotation(15))

    if config.get('use_colorjitter', False):
        # Enhanced color jittering based on analysis
        transform_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

    if config.get('use_randaugment', False):
        # Add RandAugment
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

    if config.get('use_trivialaugment', False):
        # Add TrivialAugmentWide (alternative to RandAugment)
        transform_list.append(transforms.TrivialAugmentWide())

    transform_list.extend([
        transforms.ToTensor(),
        normalize
    ])

    return transforms.Compose(transform_list)

# --- CutMix --- (From Claude_v2.py)
def cutmix(data, targets, alpha=1.0, num_classes=10):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    # Adjust targets
    targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
    shuffled_targets_onehot = F.one_hot(shuffled_targets, num_classes=num_classes).float()

    # Area of the patch
    patch_area = (x1 - x0) * (y1 - y0)
    if image_h * image_w > 0:
        lam = 1 - (patch_area / (image_h * image_w))
    else:
        lam = 1.0 # Avoid division by zero if image size is 0

    mixed_targets = lam * targets_onehot + (1 - lam) * shuffled_targets_onehot

    return data, mixed_targets

# --- Mixup --- (New augmentation method)
def mixup(data, targets, alpha=1.0, num_classes=10):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    # Mix images
    mixed_data = lam * data + (1 - lam) * shuffled_data

    # Mix targets (one-hot encoded)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).float()
    shuffled_targets_onehot = F.one_hot(shuffled_targets, num_classes=num_classes).float()
    mixed_targets = lam * targets_onehot + (1 - lam) * shuffled_targets_onehot

    return mixed_data, mixed_targets


def load_data(config):
    """Loads CIFAR-10 dataset based on configuration."""
    train_transform = get_transforms(config)
    test_transform = get_transforms({'use_augmentation': False}) # No augmentation for test set

    # Training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True
    )

    # Test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True
    )

    return trainloader, testloader

# =============================================================================
# 2. CNN Model Design and Implementation
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual Block with pre-activation."""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) # Apply shortcut to activated input if needed
        # shortcut = self.shortcut(x) # Original ResNet applies shortcut to input 'x'
        out = self.conv1(out)
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out

class AdvancedCNN(nn.Module):
    """Advanced CNN model with Residual Blocks and optional Global Average Pooling."""
    def __init__(self, config):
        super(AdvancedCNN, self).__init__()
        self.in_channels = config.get('initial_channels', 64)
        self.use_gap = config.get('use_gap', True)
        dropout_rate = config.get('dropout_rate', 0.3)
        block_config = config.get('block_config', [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)]) # (channels, num_blocks, stride)

        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # Removed initial BN and ReLU to use pre-activation blocks

        # Residual layers
        layers = []
        for out_channels, num_blocks, stride in block_config:
            layers.append(self._make_layer(out_channels, num_blocks, stride, dropout_rate))
        self.residual_layers = nn.Sequential(*layers)

        # Final Batch Norm and ReLU after blocks
        self.final_bn = nn.BatchNorm2d(block_config[-1][0]) # Channels of the last block

        # Classifier
        if self.use_gap:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(block_config[-1][0], 10) # GAP output size matches last block channels
        else:
            # Calculate input size for linear layer if not using GAP
            # Assuming input 32x32, final feature map size needs calculation based on strides
            final_size = 32
            for _, _, stride in block_config:
                final_size //= stride
            linear_input_size = block_config[-1][0] * final_size * final_size
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(linear_input_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), # Add dropout here as well if not using GAP
                nn.Linear(512, 10)
            )

    def _make_layer(self, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        # First block handles stride and channel changes
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, dropout_rate))
        self.in_channels = out_channels # Update in_channels for subsequent blocks
        # Remaining blocks in the layer
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels, 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual_layers(out)
        out = F.relu(self.final_bn(out)) # Apply final activation
        out = self.avg_pool(out) if self.use_gap else out
        out = out.view(out.size(0), -1) if self.use_gap else out # Flatten GAP output
        out = self.fc(out)
        return out

# =============================================================================
# 3. Model Training and Optimization
# =============================================================================

# --- Label Smoothing --- (From Claude_v2.py)
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, num_classes=10):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        # pred: (N, C), target: (N)
        pred = F.log_softmax(pred, dim=-1)
        with torch.no_grad():
            # Create smooth labels
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        # Calculate KL divergence loss
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# --- Warmup Scheduler --- (New)
class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr + alpha * (self.target_lr - base_lr) for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                # Ensure correct LR is set for the after_scheduler right after warmup
                for group, lr in zip(self.optimizer.param_groups, self.get_warmup_end_lr()): group['lr'] = lr
                self.finished_warmup = True

            # If there's a scheduler to use after warmup, rely on it
            if self.after_scheduler:
                 # Need to get the LR from the after_scheduler
                 # Note: This assumes after_scheduler.get_last_lr() exists and is correct
                 # A potential issue: after_scheduler's internal state might need adjustment
                 # based on the number of epochs already passed during warmup.
                 # For simplicity here, we just return its current LR.
                 # A more robust implementation might adjust after_scheduler's last_epoch.
                 try:
                     return self.after_scheduler.get_last_lr()
                 except AttributeError:
                     # Fallback if get_last_lr isn't standard
                     return [group['lr'] for group in self.optimizer.param_groups]
            else:
                # If no after_scheduler, just keep the target LR
                return [self.target_lr] * len(self.optimizer.param_groups)

    def get_warmup_end_lr(self):
         return [self.target_lr] * len(self.base_lrs)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.finished_warmup and self.after_scheduler:
            # Step the after_scheduler, handling potential epoch arguments needed by some schedulers
            if isinstance(self.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 # ReduceLROnPlateau needs a metric, cannot be stepped directly here
                 # It should be stepped in the training loop based on validation loss
                 pass
            else:
                 self.after_scheduler.step(epoch - self.warmup_epochs)
                 # Adjust epoch for the after_scheduler
        else:
            # Update LR based on warmup logic
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

def get_optimizer(model, config):
    optimizer_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 1e-4)
    momentum = config.get('momentum', 0.9)

    if optimizer_name == 'adamw':
        print(f"Using AdamW optimizer with LR={lr}, WD={weight_decay}")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        print(f"Using Adam optimizer with LR={lr}, WD={weight_decay}")
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        use_nesterov = config.get('use_nesterov', True)
        print(f"Using SGD optimizer with LR={lr}, Momentum={momentum}, WD={weight_decay}, Nesterov={use_nesterov}")
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=use_nesterov)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config, steps_per_epoch):
    scheduler_name = config.get('scheduler', 'cosine').lower()
    epochs = config['epochs']
    base_lr = optimizer.param_groups[0]['lr'] # Initial LR set in optimizer
    max_lr = config.get('max_lr', base_lr * 10) # For OneCycleLR
    warmup_epochs = config.get('warmup_epochs', 0)
    after_scheduler = None

    if scheduler_name == 'cosine':
        print(f"Using CosineAnnealingLR scheduler for {epochs - warmup_epochs} epochs.")
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=0)
    elif scheduler_name == 'onecycle':
        print(f"Using OneCycleLR scheduler with max_lr={max_lr}.")
        after_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=epochs * steps_per_epoch,
            pct_start=config.get('onecycle_pct_start', 0.3),
            anneal_strategy='cos'
        )
    elif scheduler_name == 'reduce':
        patience = config.get('scheduler_patience', 10)
        factor = config.get('scheduler_factor', 0.1)
        print(f"Using ReduceLROnPlateau scheduler with patience={patience}, factor={factor}.")
        # Note: ReduceLROnPlateau is stepped based on validation metric, not epoch.
        # We create it here, but the main train loop needs to handle its step.
        after_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
        # Warmup doesn't naturally compose with ReduceLROnPlateau in this structure.
        # We will return ReduceLROnPlateau directly if selected, ignoring warmup for it.
        if warmup_epochs > 0:
            print("Warning: Warmup is typically not used with ReduceLROnPlateau.")
        return after_scheduler
    elif scheduler_name == 'step':
        step_size = config.get('scheduler_step_size', 15)
        gamma = config.get('scheduler_gamma', 0.1)
        print(f"Using StepLR scheduler with step_size={step_size}, gamma={gamma}.")
        after_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'none':
        print("No LR scheduler used.")
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # Apply warmup if specified and not using ReduceLROnPlateau
    if warmup_epochs > 0 and scheduler_name != 'reduce':
        print(f"Applying {warmup_epochs} epochs of linear warmup.")
        # Start warmup from a very small LR or the base_lr if preferred
        initial_warmup_lr = base_lr / 100
        for group in optimizer.param_groups: group['lr'] = initial_warmup_lr
        warmup_scheduler = WarmupLR(optimizer, warmup_epochs, target_lr=base_lr, after_scheduler=after_scheduler)
        return warmup_scheduler
    else:
        return after_scheduler

def train_model(model, trainloader, optimizer, scheduler, criterion, config):
    model.to(device)
    epochs = config['epochs']
    use_cutmix = config.get('use_cutmix', False)
    use_mixup = config.get('use_mixup', False)
    cutmix_alpha = config.get('cutmix_alpha', 1.0)
    mixup_alpha = config.get('mixup_alpha', 1.0)

    # Ensure only one of cutmix or mixup is active per batch
    if use_cutmix and use_mixup:
        print("Warning: Both CutMix and Mixup enabled. Will randomly choose one per batch.")

    train_losses = []
    train_accs = []
    lrs = []

    steps_per_epoch = len(trainloader)

    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            current_batch_size = inputs.size(0)

            apply_cutmix = use_cutmix and random.random() > 0.5
            apply_mixup = use_mixup and (not apply_cutmix) and random.random() > 0.5

            if apply_cutmix:
                inputs, targets_mixed = cutmix(inputs, labels, cutmix_alpha)
                outputs = model(inputs)
                # CutMix/Mixup loss: Sum over class dimension, mean over batch dimension
                loss = -torch.sum(targets_mixed * F.log_softmax(outputs, dim=1), dim=1).mean()
            elif apply_mixup:
                inputs, targets_mixed = mixup(inputs, labels, mixup_alpha)
                outputs = model(inputs)
                loss = -torch.sum(targets_mixed * F.log_softmax(outputs, dim=1), dim=1).mean()
            else:
                # Regular training
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient Clipping
            # if config.get('grad_clip', 0) > 0:
            #     nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

            running_loss += loss.item() * current_batch_size

            # Accuracy calculation (use original labels for CutMix/Mixup accuracy estimate)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if apply_cutmix or apply_mixup:
                 # Accuracy for mixed samples is tricky, approximate using highest prob from mixed target
                 _, true_labels = torch.max(targets_mixed, 1)
                 correct += (predicted == true_labels).sum().item()
            else:
                 correct += (predicted == labels).sum().item()

            # Step OneCycleLR or schedulers that step per batch
            if isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, WarmupLR)):
                 if not (isinstance(scheduler, WarmupLR) and scheduler.finished_warmup and isinstance(scheduler.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                    scheduler.step()
                 lrs.append(optimizer.param_groups[0]['lr']) # Log LR per step

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        current_lr = optimizer.param_groups[0]['lr']
        if not isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, WarmupLR)):
             lrs.append(current_lr) # Log LR per epoch

        # Step LR scheduler if it's epoch-based
        if scheduler is not None and not isinstance(scheduler, (torch.optim.lr_scheduler.OneCycleLR, torch.optim.lr_scheduler.ReduceLROnPlateau, WarmupLR)):
            scheduler.step()
        # Special handling for ReduceLROnPlateau inside Warmup wrapper
        elif isinstance(scheduler, WarmupLR) and scheduler.finished_warmup and isinstance(scheduler.after_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
             # Requires validation loss, needs to be passed from main loop if validation is done
             # scheduler.step(val_loss) # Placeholder
             pass # Step ReduceLROnPlateau based on validation loss in main loop
        # Direct handling for ReduceLROnPlateau (if not wrapped)
        elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
             # scheduler.step(val_loss) # Placeholder
             pass # Step based on validation loss in main loop


        end_time = time.time()
        print(f'[Epoch: {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, '
              f'LR: {current_lr:.6f}, Time: {end_time - start_time:.2f}s')

    # If OneCycleLR or Warmup was used, lrs list contains per-step LRs.
    # We might want to return per-epoch LRs instead/as well for plotting consistency.
    # For now, returning the potentially per-step LRs.
    return train_losses, train_accs, lrs

# =============================================================================
# 4. Quantitative Evaluation and Result Analysis
# =============================================================================

def evaluate_model(model, testloader, config):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Test Set Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

# --- Visualization Functions --- (Adapted from Claude_v2 and Gemini_v2)

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.show()

def plot_training_curves(results_dict, title="Training Process Comparison"):
    num_plots = len(results_dict)
    plt.figure(figsize=(15, 5 * num_plots)) # Adjust figure size based on number of plots
    plot_index = 1

    for name, data in results_dict.items():
        epochs = len(data['losses'])
        # Loss Curve
        plt.subplot(num_plots, 3, plot_index)
        plt.plot(range(epochs), data['losses'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name} - Loss')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plot_index += 1

        # Accuracy Curve
        plt.subplot(num_plots, 3, plot_index)
        plt.plot(range(epochs), data['accs'], label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{name} - Accuracy')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plot_index += 1

        # Learning Rate Curve
        plt.subplot(num_plots, 3, plot_index)
        # Check if LR is per step or per epoch
        if len(data.get('lrs', [])) > epochs:
             # Per-step LR (e.g., OneCycleLR, Warmup)
             steps = len(data['lrs'])
             plt.plot(np.linspace(0, epochs, steps), data['lrs'], label=name)
             plt.xlabel('Epoch (interpolated for steps)')
        else:
             # Per-epoch LR
             plt.plot(range(len(data.get('lrs', []))), data['lrs'], label=name)
             plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title(f'{name} - Learning Rate')
        plt.grid(True, alpha=0.5)
        plt.legend()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plot_index += 1

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

def compare_final_accuracies(accuracies, title='Final Test Accuracy Comparison'):
    plt.figure(figsize=(10, 6))
    names = list(accuracies.keys())
    values = list(accuracies.values())
    bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0, 1, len(accuracies))))
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.ylim(max(0, min(values) - 2), min(100, max(values) + 2)) # Adjust y-limits
    for i, v in enumerate(values):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center', va='bottom')
    plt.xticks(rotation=15, ha='right') # Rotate labels if they overlap
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. Main Execution Logic - Experiment Runner
# =============================================================================

def main():
    print("\n=== Gemini V3: Advanced CNN Experiments for CIFAR-10 ===")

    # --- Define Experiment Configurations --- #
    experiment_configs = {
        # Baseline similar to Gemini_v2 SGD run, but using AdvancedCNN (ResBlocks)
        'SGD_StepLR_Baseline': {
            'model_name': 'AdvancedCNN',
            'use_gap': True,
            'dropout_rate': 0.3,
            'initial_channels': 64,
            'block_config': [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)],
            'optimizer': 'sgd',
            'learning_rate': 0.1, # Higher initial LR common for SGD
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'use_nesterov': True,
            'scheduler': 'step',
            'scheduler_step_size': 15,
            'scheduler_gamma': 0.1,
            'epochs': 50,
            'batch_size': 128,
            'use_augmentation': True,
            'use_rotation': True,
            'use_colorjitter': True,
            'use_cutmix': False,
            'use_mixup': False,
            'label_smoothing': 0.0,
            'warmup_epochs': 0
        },
        # AdamW + Cosine Annealing + Advanced Augmentation
        'AdamW_Cosine_AdvAug': {
            'model_name': 'AdvancedCNN',
            'use_gap': True,
            'dropout_rate': 0.4,
            'initial_channels': 64,
            'block_config': [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)],
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01, # WD often higher for AdamW
            'scheduler': 'cosine',
            'epochs': 75, # Longer training often beneficial with cosine
            'batch_size': 128,
            'use_augmentation': True,
            'use_rotation': True,
            'use_colorjitter': True,
            'use_randaugment': True, # Add RandAugment
            'use_cutmix': True,      # Add CutMix
            'cutmix_alpha': 1.0,
            'use_mixup': False,
            'label_smoothing': 0.1,  # Add Label Smoothing
            'warmup_epochs': 5
        },
        # SGD + OneCycleLR + Advanced Augmentation
        'SGD_OneCycle_AdvAug': {
            'model_name': 'AdvancedCNN',
            'use_gap': True,
            'dropout_rate': 0.4,
            'initial_channels': 64,
            'block_config': [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)],
            'optimizer': 'sgd',
            'learning_rate': 0.01, # Base LR for OneCycle (will ramp up)
            'max_lr': 0.1,        # Max LR for OneCycle
            'momentum': 0.9,
            'weight_decay': 5e-4,
            'use_nesterov': True,
            'scheduler': 'onecycle',
            'epochs': 50,
            'batch_size': 128,
            'use_augmentation': True,
            'use_rotation': True,
            'use_colorjitter': True,
            'use_trivialaugment': True, # Try TrivialAugment
            'use_cutmix': False,
            'use_mixup': True,       # Try Mixup instead of CutMix
            'mixup_alpha': 0.8,
            'label_smoothing': 0.1,
            'warmup_epochs': 0 # OneCycle has its own warmup
        },
        # Wider Network Example (AdamW + Cosine)
        'AdamW_Cosine_WiderNet': {
            'model_name': 'AdvancedCNN',
            'use_gap': True,
            'dropout_rate': 0.5, # Increase dropout for wider net
            'initial_channels': 96, # Wider start
            'block_config': [(96, 2, 1), (192, 2, 2), (384, 2, 2), (768, 2, 2)], # Wider blocks
            'optimizer': 'adamw',
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'scheduler': 'cosine',
            'epochs': 75,
            'batch_size': 128,
            'use_augmentation': True,
            'use_rotation': True,
            'use_colorjitter': True,
            'use_randaugment': True,
            'use_cutmix': True,
            'cutmix_alpha': 1.0,
            'label_smoothing': 0.1,
            'warmup_epochs': 5
        },
    }

    # --- Run Experiments --- #
    all_run_results = {}
    final_accuracies = {}
    confusion_matrices = {}

    for name, config in experiment_configs.items():
        print(f"\n=== Running Experiment: {name} ===")
        print(f"Config: {config}")

        # Load Data
        trainloader, testloader = load_data(config)

        # Create Model
        model = AdvancedCNN(config).to(device)
        print(f"Model: {config['model_name']}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


        # Create Criterion
        if config.get('label_smoothing', 0.0) > 0:
            print(f"Using Label Smoothing: {config['label_smoothing']}")
            criterion = LabelSmoothingLoss(smoothing=config['label_smoothing'])
        else:
            criterion = nn.CrossEntropyLoss()

        # Create Optimizer
        optimizer = get_optimizer(model, config)

        # Create Scheduler
        steps_per_epoch = len(trainloader)
        scheduler = get_scheduler(optimizer, config, steps_per_epoch)

        # Train Model
        train_losses, train_accs, lrs = train_model(
            model, trainloader, optimizer, scheduler, criterion, config
        )

        # Evaluate Model
        accuracy, cm = evaluate_model(model, testloader, config)

        # Store Results
        all_run_results[name] = {'losses': train_losses, 'accs': train_accs, 'lrs': lrs}
        final_accuracies[name] = accuracy
        confusion_matrices[name] = cm

        # Optional: Save model checkpoint
        # torch.save(model.state_dict(), f'{name}_model.pth')

    # --- Analyze and Visualize Results --- #
    print("\n=== Experiment Results Summary ===")

    # Final Accuracy Comparison
    compare_final_accuracies(final_accuracies, title='Final Test Accuracy Comparison Across Experiments')

    # Training Curves
    plot_training_curves(all_run_results, title='Training Process Comparison')

    # Confusion Matrices
    for name, cm in confusion_matrices.items():
        plot_confusion_matrix(cm, title=f'Confusion Matrix - {name}')

    print("\n=== All Experiments Completed ===")
    for name, acc in final_accuracies.items():
        print(f"- {name}: {acc:.2f}%")

if __name__ == "__main__":
    main() 