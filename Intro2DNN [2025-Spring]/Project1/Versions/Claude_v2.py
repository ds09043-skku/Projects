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

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device in use: {device}')

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
BASE_LR = 0.001
MAX_LR = 0.01
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5  # Increased dropout rate
LABEL_SMOOTHING = 0.1  # Added label smoothing

# Data preprocessing and augmentation
def get_transforms(use_augmentation=False):
    # Basic transform - normalization only
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 statistics
    ])
    
    if not use_augmentation:
        return base_transform
    
    # Enhanced data augmentation transforms
    augmentation_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        # Enhanced color jittering
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        # Added RandAugment - applies various augmentations randomly
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return augmentation_transform

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

# Load data
def load_data(use_augmentation=False, use_cutmix=False):
    transform = get_transforms(use_augmentation)
    
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

# Residual block definition
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (if channel size or dimensions change)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Residual connection
        out = F.relu(out)
        return out

# Improved CNN model definition
class ImprovedCNN(nn.Module):
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual block layers
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 10)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
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

# Model training function
def train_model(model, trainloader, optimizer, scheduler, criterion, epochs, use_cutmix=False):
    model.to(device)
    
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        model.train()
        for i, data in enumerate(trainloader, 0):
            if use_cutmix and np.random.random() > 0.5:
                # Apply CutMix
                inputs, targets_mixed = cutmix(data)
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
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        # Learning rate scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(running_loss / total)
            elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                # OneCycleLR should be called per batch, but here we're calling per epoch
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
    
    return train_losses, train_accs

# Model evaluation function
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

def main():
    print("\n=== CNN Model Performance Enhancement Experiment for CIFAR-10 Classification ===")
    
    # Results storage variables
    all_results = {}
    final_accuracies = {}
    all_confusion_matrices = {}
    
    # 1. Baseline model (Data Augmentation + Adam + ReduceLROnPlateau)
    print("\n=== Experiment 1: Baseline Model (Data Augmentation + Adam + ReduceLROnPlateau) ===")
    trainloader_aug, testloader = load_data(use_augmentation=True)
    
    baseline_model = ImprovedCNN(dropout_rate=0.3).to(device)  # Base dropout rate
    criterion = nn.CrossEntropyLoss()
    optimizer_adam = optim.Adam(baseline_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=5, verbose=True)
    
    baseline_losses, baseline_accs = train_model(
        baseline_model, trainloader_aug, optimizer_adam, scheduler_adam, criterion, epochs=EPOCHS
    )
    baseline_accuracy, baseline_cm = evaluate_model(baseline_model, testloader)
    
    all_results['Baseline Model'] = {'losses': baseline_losses, 'accs': baseline_accs}
    final_accuracies['Baseline Model'] = baseline_accuracy
    all_confusion_matrices['Baseline Model'] = baseline_cm
    
    # 2. Enhanced model (Extended Data Augmentation + One-Cycle LR)
    print("\n=== Experiment 2: Enhanced Model (Extended Data Augmentation + One-Cycle LR) ===")
    enhanced_trainloader, _ = load_data(use_augmentation=True)
    
    enhanced_model = ImprovedCNN(dropout_rate=DROPOUT_RATE).to(device)  # Higher dropout rate
    
    # Loss function with label smoothing
    criterion_smooth = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Adam optimizer with One-Cycle LR policy
    optimizer_enhanced = optim.Adam(enhanced_model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler_one_cycle = optim.lr_scheduler.OneCycleLR(
        optimizer_enhanced, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(enhanced_trainloader)
    )
    
    enhanced_losses, enhanced_accs = train_model(
        enhanced_model, enhanced_trainloader, optimizer_enhanced, scheduler_one_cycle, 
        criterion_smooth, epochs=EPOCHS, use_cutmix=True
    )
    enhanced_accuracy, enhanced_cm = evaluate_model(enhanced_model, testloader)
    
    all_results['Enhanced Model'] = {'losses': enhanced_losses, 'accs': enhanced_accs}
    final_accuracies['Enhanced Model'] = enhanced_accuracy
    all_confusion_matrices['Enhanced Model'] = enhanced_cm
    
    # 3. Result comparison and visualization
    print("\n=== Result Comparison and Visualization ===")
    
    # Final accuracy comparison
    compare_results(
        [final_accuracies['Baseline Model'], final_accuracies['Enhanced Model']],
        ['Baseline Model', 'Enhanced Model']
    )
    
    # Training process visualization
    plot_training_curves(all_results, title='Model Training Process Comparison')
    
    # Confusion matrix visualization
    print("\nBaseline Model - Confusion Matrix:")
    plot_confusion_matrix(all_confusion_matrices['Baseline Model'], title='Baseline Model Confusion Matrix')
    
    print("\nEnhanced Model - Confusion Matrix:")
    plot_confusion_matrix(all_confusion_matrices['Enhanced Model'], title='Enhanced Model Confusion Matrix')
    
    print("\n=== All Experiments Completed ===")
    print(f"Baseline Model Final Accuracy: {final_accuracies['Baseline Model']:.2f}%")
    print(f"Enhanced Model Final Accuracy: {final_accuracies['Enhanced Model']:.2f}%")
    print(f"Accuracy Improvement: {final_accuracies['Enhanced Model'] - final_accuracies['Baseline Model']:.2f}%p")

if __name__ == "__main__":
    main() 
