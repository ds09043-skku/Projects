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
import math
import random
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.swa_utils import AveragedModel, update_bn

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device in use: {device}')

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 75  # Increased epoch count
BASE_LR = 0.001
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5
LABEL_SMOOTHING = 0.1

# =============================================================================
# 1. Enhanced Data Preprocessing and Augmentation
# =============================================================================

def get_transforms(use_augmentation=False, rand_augment=False):
    # Basic transform - normalization only
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 statistics
    ])
    
    if not use_augmentation:
        return base_transform
    
    # Enhanced data augmentation transforms
    transform_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Enhanced color jittering
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)
    ]
    
    # Add RandAugment if specified
    if rand_augment:
        transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return transforms.Compose(transform_list)

# CutMix data augmentation function
def cutmix(batch, alpha=1.0):
    data, targets = batch
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
    
    # Adjusted targets
    targets_onehot = F.one_hot(targets, num_classes=10).float()
    shuffled_targets_onehot = F.one_hot(shuffled_targets, num_classes=10).float()
    
    lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))
    mixed_targets = lam * targets_onehot + (1 - lam) * shuffled_targets_onehot
    
    return data, mixed_targets

# MixUp data augmentation function (new)
def mixup(batch, alpha=1.0):
    data, targets = batch
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    
    lam = np.random.beta(alpha, alpha)
    mixed_data = lam * data + (1 - lam) * shuffled_data
    
    # Create one-hot encoded targets
    targets_onehot = F.one_hot(targets, num_classes=10).float()
    shuffled_targets_onehot = F.one_hot(shuffled_targets, num_classes=10).float()
    
    mixed_targets = lam * targets_onehot + (1 - lam) * shuffled_targets_onehot
    
    return mixed_data, mixed_targets

# GridMask augmentation (new)
def grid_mask(img, p=0.5, d_ratio=0.5):
    if random.random() > p:
        return img
    
    h, w = img.shape[-2:]
    d = int(d_ratio * min(h, w))
    
    mask = torch.ones_like(img)
    for i in range(0, h, d):
        for j in range(0, w, d):
            if i + d//2 < h and j + d//2 < w:
                mask[:, :, i:i+d//2, j:j+d//2] = 0
                mask[:, :, i+d//2:min(i+d, h), j+d//2:min(j+d, w)] = 0
    
    masked_img = img * mask
    return masked_img

# Load data
def load_data(use_augmentation=False, use_cutmix=False, rand_augment=False):
    transform = get_transforms(use_augmentation, rand_augment)
    
    # Training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    
    # Test dataset (no augmentation)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, 
        transform=get_transforms(use_augmentation=False)
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return trainloader, testloader

# =============================================================================
# 2. Advanced CNN Model Design and Implementation
# =============================================================================

# Squeeze and Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Efficient Channel Attention Module
class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs(math.log(channel, 2) + b) / gamma)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = y.unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# DropBlock for more structured dropout
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        
    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        
        gamma = self.drop_prob / (self.block_size ** 2)
        mask_shape = x.shape
        
        # Sample mask
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        
        # Apply block mask
        block_mask = self._compute_block_mask(mask)
        
        # Scale output
        out = x * block_mask[:, :, :, :]
        out = out * block_mask.numel() / block_mask.sum()
        
        return out
    
    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(mask, 
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)
        
        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
            
        block_mask = 1 - block_mask
        return block_mask

# Enhanced Residual Block with SE/ECA module option
class EnhancedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, attention_type='se', drop_block=False):
        super(EnhancedResBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Attention mechanism
        if attention_type == 'se':
            self.attention = SEBlock(out_channels)
        elif attention_type == 'eca':
            self.attention = ECABlock(out_channels)
        else:
            self.attention = None
        
        # DropBlock
        self.use_drop_block = drop_block
        if drop_block:
            self.drop_block = DropBlock2D(drop_prob=0.1, block_size=5)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
            
        if self.use_drop_block and self.training:
            out = self.drop_block(out)
            
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out

# Advanced CNN with attention mechanisms and structured dropout
class AdvancedCNN(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE, attention_type='se', use_drop_block=False):
        super(AdvancedCNN, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual block layers with attention mechanism
        self.layer1 = self._make_layer(64, 64, 2, stride=1, attention_type=attention_type, drop_block=use_drop_block)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, attention_type=attention_type, drop_block=use_drop_block)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, attention_type=attention_type, drop_block=use_drop_block)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, attention_type=attention_type, drop_block=use_drop_block)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, attention_type, drop_block):
        layers = []
        layers.append(EnhancedResBlock(in_channels, out_channels, stride, attention_type, drop_block))
        for _ in range(1, num_blocks):
            layers.append(EnhancedResBlock(out_channels, out_channels, 1, attention_type, drop_block))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = self.fc2(out)
        
        return out

# =============================================================================
# 3. Advanced Training and Optimization Strategies
# =============================================================================

# Cosine Annealing with Warm Restarts
class CosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.cycle = 0
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch == -1:
            return self.base_lrs
        elif self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_0) % (self.T_i) == 0:
            self.cycle += 1
            self.T_i = self.T_0 * (self.T_mult ** self.cycle)
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi)) / 2
                    for base_lr in self.base_lrs]
        
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * (self.last_epoch - 1) % self.T_i / self.T_i)) / 2
                for base_lr in self.base_lrs]

# Advanced model training function with augmentation options
def train_model(model, trainloader, optimizer, scheduler, criterion, epochs, 
                use_cutmix=False, use_mixup=False, use_grid_mask=False, swa_model=None, swa_start=None):
    model.to(device)
    if swa_model:
        swa_model.to(device)
    
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for i, data in enumerate(trainloader, 0):
            # Decide which augmentation to use
            use_this_cutmix = use_cutmix and np.random.random() > 0.5
            use_this_mixup = not use_this_cutmix and use_mixup and np.random.random() > 0.5
            
            if use_this_cutmix:
                # Apply CutMix
                inputs, targets_mixed = cutmix(data)
                inputs = inputs.to(device)
                targets_mixed = targets_mixed.to(device)
                
                if use_grid_mask:
                    inputs = grid_mask(inputs)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.sum(targets_mixed * F.log_softmax(outputs, dim=1), dim=1).mean().neg()
                
                # Extract class with highest probability from one-hot encoding
                _, predicted = torch.max(outputs.data, 1)
                _, true_targets = torch.max(targets_mixed, 1)
                total += true_targets.size(0)
                correct += (predicted == true_targets).sum().item()
            
            elif use_this_mixup:
                # Apply MixUp
                inputs, targets_mixed = mixup(data)
                inputs = inputs.to(device)
                targets_mixed = targets_mixed.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.sum(targets_mixed * F.log_softmax(outputs, dim=1), dim=1).mean().neg()
                
                # Extract class with highest probability from one-hot encoding
                _, predicted = torch.max(outputs.data, 1)
                _, true_targets = torch.max(targets_mixed, 1)
                total += true_targets.size(0)
                correct += (predicted == true_targets).sum().item()
            
            else:
                # Regular training
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                if use_grid_mask:
                    inputs = grid_mask(inputs)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        # SWA model update
        if swa_model and epoch >= swa_start:
            swa_model.update_parameters(model)
            
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss / total)
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                # OneCycleLR should be called per batch, but here we're simplifying
                pass
            else:
                scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        end_time = time.time()
        print(f'[Epoch: {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, '
              f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}, Time: {end_time - start_time:.2f}s')
    
    # Update BatchNorm statistics for SWA model
    if swa_model:
        update_bn(trainloader, swa_model, device=device)
    
    return train_losses, train_accs

# =============================================================================
# 4. Model Evaluation and Analysis
# =============================================================================

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'Test set accuracy: {accuracy:.2f}%')
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

# Confusion matrix visualization function
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.show()

# Model performance comparison function
def compare_results(accuracies, model_names):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(accuracies))))
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Training process visualization function
def plot_training_curves(results_dict, title="Training Process"):
    plt.figure(figsize=(15, 6))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    for name, data in results_dict.items():
        plt.plot(data['losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    for name, data in results_dict.items():
        plt.plot(data['accs'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def class_accuracy(confusion_matrix):
    """Calculate per-class accuracy from confusion matrix."""
    per_class_acc = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    return per_class_acc * 100

def plot_class_accuracies(confusion_matrices, model_names):
    """Plot per-class accuracies for different models."""
    plt.figure(figsize=(12, 8))
    
    for i, (name, cm) in enumerate(zip(model_names, confusion_matrices)):
        per_class_acc = class_accuracy(cm)
        plt.plot(range(10), per_class_acc, marker='o', label=name)
    
    plt.xticks(range(10), classes)
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Per-Class Accuracy Comparison')
    plt.grid(True, alpha=0.3)
    plt.show()

# =============================================================================
# 5. Main Execution and Experiments
# =============================================================================

def main():
    print("\n=== Advanced CNN with Attention Mechanisms for CIFAR-10 Classification ===")
    
    # Results storage variables
    all_results = {}
    final_accuracies = {}
    all_confusion_matrices = {}
    
    # 1. Baseline model (similar to Improved Model from Claude_v2)
    print("\n=== Experiment 1: Baseline Model (Enhanced from Claude_v2) ===")
    trainloader_aug, testloader = load_data(use_augmentation=True, rand_augment=True)
    
    baseline_model = AdvancedCNN(dropout_rate=DROPOUT_RATE, attention_type=None).to(device)
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer_adam = optim.Adam(baseline_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_one_cycle = optim.lr_scheduler.OneCycleLR(
        optimizer_adam, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(trainloader_aug)
    )
    
    baseline_losses, baseline_accs = train_model(
        baseline_model, trainloader_aug, optimizer_adam, scheduler_one_cycle, criterion_smooth, 
        epochs=EPOCHS, use_cutmix=True
    )
    baseline_accuracy, baseline_cm = evaluate_model(baseline_model, testloader)
    
    all_results['Baseline Model'] = {'losses': baseline_losses, 'accs': baseline_accs}
    final_accuracies['Baseline Model'] = baseline_accuracy
    all_confusion_matrices['Baseline Model'] = baseline_cm
    
    # 2. SE-Net model (Adding Squeeze-and-Excitation)
    print("\n=== Experiment 2: SE-Net Model (Adding Squeeze-and-Excitation) ===")
    
    se_model = AdvancedCNN(dropout_rate=DROPOUT_RATE, attention_type='se').to(device)
    optimizer_adamw = optim.AdamW(se_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=EPOCHS, eta_min=0)
    
    se_losses, se_accs = train_model(
        se_model, trainloader_aug, optimizer_adamw, scheduler_cosine, criterion_smooth, 
        epochs=EPOCHS, use_cutmix=True, use_mixup=True
    )
    se_accuracy, se_cm = evaluate_model(se_model, testloader)
    
    all_results['SE-Net Model'] = {'losses': se_losses, 'accs': se_accs}
    final_accuracies['SE-Net Model'] = se_accuracy
    all_confusion_matrices['SE-Net Model'] = se_cm
    
    # 3. ECA-Net model (Efficient Channel Attention)
    print("\n=== Experiment 3: ECA-Net Model (Efficient Channel Attention) ===")
    
    eca_model = AdvancedCNN(dropout_rate=DROPOUT_RATE, attention_type='eca').to(device)
    optimizer_adamw_eca = optim.AdamW(eca_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    
    # Custom warm restart scheduler
    custom_scheduler = CosineAnnealingWarmRestarts(optimizer_adamw_eca, T_0=10, T_mult=2, eta_min=1e-6)
    
    eca_losses, eca_accs = train_model(
        eca_model, trainloader_aug, optimizer_adamw_eca, custom_scheduler, criterion_smooth, 
        epochs=EPOCHS, use_cutmix=False, use_mixup=True, use_grid_mask=True
    )
    eca_accuracy, eca_cm = evaluate_model(eca_model, testloader)
    
    all_results['ECA-Net Model'] = {'losses': eca_losses, 'accs': eca_accs}
    final_accuracies['ECA-Net Model'] = eca_accuracy
    all_confusion_matrices['ECA-Net Model'] = eca_cm
    
    # 4. Advanced model with SWA (Stochastic Weight Averaging)
    print("\n=== Experiment 4: Advanced Model with SWA (Stochastic Weight Averaging) ===")
    
    swa_base_model = AdvancedCNN(dropout_rate=DROPOUT_RATE, attention_type='se', use_drop_block=True).to(device)
    swa_model = AveragedModel(swa_base_model)
    
    optimizer_adamw_swa = optim.AdamW(swa_base_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_cosine_swa = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw_swa, T_max=EPOCHS, eta_min=0)
    
    swa_losses, swa_accs = train_model(
        swa_base_model, trainloader_aug, optimizer_adamw_swa, scheduler_cosine_swa, criterion_smooth, 
        epochs=EPOCHS, use_cutmix=True, use_mixup=True, use_grid_mask=True,
        swa_model=swa_model, swa_start=int(EPOCHS * 0.75)  # Start SWA at 75% of training
    )
    
    # Evaluate SWA model
    swa_accuracy, swa_cm = evaluate_model(swa_model, testloader)
    
    all_results['SWA Model'] = {'losses': swa_losses, 'accs': swa_accs}
    final_accuracies['SWA Model'] = swa_accuracy
    all_confusion_matrices['SWA Model'] = swa_cm
    
    # Result comparison and visualization
    print("\n=== Result Comparison and Visualization ===")
    
    # Final accuracy comparison
    compare_results(
        [final_accuracies[key] for key in final_accuracies.keys()],
        list(final_accuracies.keys())
    )
    
    # Training process visualization
    plot_training_curves(all_results, title='Model Training Process Comparison')
    
    # Confusion matrix visualization for best model
    best_model = max(final_accuracies.items(), key=lambda x: x[1])[0]
    print(f"\nBest Model: {best_model} with accuracy {final_accuracies[best_model]:.2f}%")
    plot_confusion_matrix(all_confusion_matrices[best_model], title=f'{best_model} Confusion Matrix')
    
    # Per-class accuracy comparison
    plot_class_accuracies(
        [all_confusion_matrices[key] for key in all_confusion_matrices.keys()],
        list(all_confusion_matrices.keys())
    )
    
    print("\n=== All Experiments Completed ===")
    for name, acc in final_accuracies.items():
        print(f"{name} Final Accuracy: {acc:.2f}%")
    print(f"Best accuracy improvement: {max(final_accuracies.values()) - baseline_accuracy:.2f}%p")

if __name__ == "__main__":
    main() 