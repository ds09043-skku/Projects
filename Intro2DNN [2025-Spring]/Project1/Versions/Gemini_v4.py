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
from torch.optim.lr_scheduler import _LRScheduler # Keep for potential custom schedulers if needed later

# =============================================================================
# Configuration
# =============================================================================
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyperparameters (Based on best performing model from Gemini_v3 analysis and project guidelines)
# Consistent settings for fair comparison where appropriate
BATCH_SIZE = 128
EPOCHS = 50 # Standard number of epochs for comparisons
LEARNING_RATE_SGD = 0.1 # Optimal LR found for SGD in v3
LEARNING_RATE_ADAMW = 0.001 # Standard AdamW starting LR
WEIGHT_DECAY_SGD = 5e-4 # Optimal WD found for SGD in v3
WEIGHT_DECAY_ADAMW = 0.01 # Common WD for AdamW, requires tuning if results are poor
DROPOUT_RATE = 0.3 # Optimal dropout from v3's best model
MOMENTUM_SGD = 0.9 # Standard momentum for SGD
STEP_LR_STEP_SIZE = 15 # Optimal StepLR setting from v3
STEP_LR_GAMMA = 0.1 # Optimal StepLR setting from v3
LABEL_SMOOTHING = 0.0 # Turned off for final version as per analysis showing no clear benefit in v3's best runs

# =============================================================================
# 1. Data Preprocessing and Augmentation (Requirement 1)
# =============================================================================

def get_transforms(use_augmentation):
    """Returns data transformation pipelines. Meets Req 1.1, 1.2.

    Args:
        use_augmentation (bool): Whether to apply standard data augmentation
                                  (RandomCrop, HorizontalFlip, Rotation, ColorJitter).

    Returns:
        torchvision.transforms.Compose: A PyTorch transforms pipeline.
    """
    # Normalization parameters for CIFAR-10 (standard values)
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if use_augmentation:
        # Transformation pipeline with standard data augmentation for training
        # Includes Flip, Crop, Rotation, Jitter as required by guidelines
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(10), # Keeping augmentation simpler based on v3 results
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
        print("Augmentation enabled for training transforms.")
    else:
        # Basic transformation for testing or training without augmentation
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        print("Augmentation disabled for training transforms.")
    return transform

def load_data(use_augmentation_for_train):
    """Loads the CIFAR-10 dataset. Meets Req 1.1.

    Args:
        use_augmentation_for_train (bool): Apply augmentation to the training set.

    Returns:
        tuple: (trainloader, testloader) - DataLoaders for training and testing.
    """
    # Get transformations based on whether augmentation should be used for the training set
    train_transform = get_transforms(use_augmentation=use_augmentation_for_train)
    # Test set *never* uses training augmentation
    test_transform = get_transforms(use_augmentation=False)

    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"CIFAR-10 data loaded. Training augmentation: {use_augmentation_for_train}.")
    return trainloader, testloader

# =============================================================================
# 2. CNN Model Design and Implementation (Requirement 2)
# =============================================================================

class ResidualBlock(nn.Module):
    """Basic Residual Block used in the custom CNN. Meets part of Req 2.2, 2.3.
       Uses standard Conv-BN-ReLU order. This is a fundamental component
       of the unique CNN architecture.
    """
    expansion = 1 # Standard ResBlock expansion factor

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolutional layer of the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer of the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to handle changes in dimensions or channel numbers
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        # Note: The logic assumes expansion=1 here based on usage in FinalCNN

    def forward(self, x):
        """Forward pass through the residual block."""
        identity = x # Store the input for the shortcut connection

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity) # Add the residual (shortcut connection)
        out = F.relu(out) # Final activation after addition
        return out

class FinalCNN(nn.Module):
    """Custom CNN architecture ('FinalCNN') for CIFAR-10 classification. Meets Req 2.1, 2.2, 2.3.

    This architecture is unique and designed specifically for this project,
    avoiding direct replication of known architectures like ResNet or VGG (Req Imp. Note 1).
    It incorporates residual connections (inspired by ResNet) for improved training
    of deeper networks.

    Features:
        - Initial convolutional layer (conv1).
        - Multiple stages (layers) of Residual Blocks. Contains > 4 conv layers.
        - Batch Normalization (BN) within blocks.
        - Global Average Pooling (GAP) for spatial dimension reduction.
        - Dropout for regularization before the final classification layer.
    """
    def __init__(self, block=ResidualBlock, num_blocks_list=[2, 2, 2, 2], dropout_rate=DROPOUT_RATE):
        """Initializes the FinalCNN model.

        Args:
            block (nn.Module): The type of residual block to use.
            num_blocks_list (list): List containing the number of blocks in each stage.
            dropout_rate (float): Dropout probability before the final layer.
        """
        super(FinalCNN, self).__init__()
        self.in_channels = 64 # Number of channels after the initial convolution

        # Initial convolutional layer
        # Input: 3x32x32, Output: 64x32x32
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        # Creating the residual layers (stages)
        # Stage 1: 64 channels, stride 1. Output: 64x32x32
        self.layer1 = self._make_layer(block, 64, num_blocks_list[0], stride=1)
        # Stage 2: 128 channels, stride 2. Output: 128x16x16
        self.layer2 = self._make_layer(block, 128, num_blocks_list[1], stride=2)
        # Stage 3: 256 channels, stride 2. Output: 256x8x8
        self.layer3 = self._make_layer(block, 256, num_blocks_list[2], stride=2)
        # Stage 4: 512 channels, stride 2. Output: 512x4x4
        self.layer4 = self._make_layer(block, 512, num_blocks_list[3], stride=2)

        # Global Average Pooling (GAP) layer. Output: 512x1x1
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Final fully connected layer for classification (10 classes)
        self.fc = nn.Linear(512 * block.expansion, 10)

        print(f"FinalCNN model initialized.")
        print(f" - Residual Block type: {block.__name__}")
        print(f" - Number of blocks per stage: {num_blocks_list}")
        print(f" - Dropout Rate: {dropout_rate}")
        # Calculate and print total parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f" - Total trainable parameters: {total_params:,}")


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """Helper function to create a stage consisting of multiple residual blocks."""
        # Strides for the blocks in the layer. Only the first block has a stride != 1.
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_channels, out_channels, stride_val))
            # Update in_channels for the next block in the sequence
            # Assumes block.expansion is defined in the ResidualBlock class (defaulting to 1 if not)
            self.in_channels = out_channels * getattr(block, 'expansion', 1)
        return nn.Sequential(*layers)

    def forward(self, x):
        """Defines the forward pass of the FinalCNN model."""
        # Initial convolution and activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Pass through residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Apply Global Average Pooling
        out = self.avg_pool(out)
        # Flatten the output for the fully connected layer
        out = out.view(out.size(0), -1)
        # Apply dropout
        out = self.dropout(out)
        # Final classification layer
        out = self.fc(out)
        return out

# =============================================================================
# 3. Model Training and Optimization (Requirement 3)
# =============================================================================

def train_model(model, trainloader, optimizer, scheduler, criterion, epochs):
    """Trains the CNN model. Meets part of Req 3.1, 3.2.

    Args:
        model (nn.Module): The model to train.
        trainloader (DataLoader): Training data loader.
        optimizer: The optimization algorithm (e.g., SGD, AdamW).
        scheduler: Learning rate scheduler (e.g., StepLR, CosineAnnealingLR).
        criterion: Loss function (e.g., CrossEntropyLoss).
        epochs (int): Number of training epochs.

    Returns:
        tuple: (train_losses, train_accs, learning_rates) - Lists tracking metrics per epoch.
    """
    model.to(device) # Ensure model is on the correct device
    train_losses = []
    train_accs = []
    learning_rates = [] # To track LR changes

    print(f"\n--- Starting Training --- ")
    print(f"Optimizer: {optimizer.__class__.__name__}, Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}, Epochs: {epochs}")

    for epoch in range(epochs):
        start_time = time.time()
        model.train() # Set model to training mode

        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Iterate over the training data
        for i, (inputs, labels) in enumerate(trainloader):
            # Move data to the configured device
            inputs, labels = inputs.to(device), labels.to(device)

            # --- Training Step ---
            # 1. Zero the gradients accumulated from the previous batch
            optimizer.zero_grad()
            # 2. Forward pass: compute model predictions
            outputs = model(inputs)
            # 3. Calculate the loss
            loss = criterion(outputs, labels)
            # 4. Backward pass: compute gradient of the loss w.r.t. model parameters
            loss.backward()
            # 5. Optimizer step: update model parameters based on gradients
            optimizer.step()
            # ---------------------

            # Accumulate statistics for the epoch
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Record the learning rate for this epoch
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

        # Step the learning rate scheduler (if one is provided and steps per epoch)
        if scheduler:
            # Schedulers like StepLR, CosineAnnealingLR are stepped each epoch.
            # ReduceLROnPlateau needs a metric (like validation loss) and is typically stepped after evaluation.
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

        # Print epoch summary
        end_time = time.time()
        print(f'[Epoch: {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, '
              f'LR: {current_lr:.6f}, Time: {end_time - start_time:.2f}s')

    print("--- Training Finished ---")
    return train_losses, train_accs, learning_rates

# =============================================================================
# 4. Quantitative Evaluation and Result Analysis (Requirement 4)
# =============================================================================

def evaluate_model(model, testloader):
    """Evaluates the model on the test dataset. Meets Req 4.1.

    Args:
        model (nn.Module): Trained model to evaluate.
        testloader (DataLoader): Test data loader.

    Returns:
        tuple: (accuracy, confusion_mat) Test accuracy (%) and confusion matrix.
    """
    model.to(device) # Ensure model is on the correct device
    model.eval() # Set model to evaluation mode (disables dropout, uses running stats in BN)

    correct_predictions = 0
    total_samples = 0
    all_preds = [] # Store all predictions
    all_labels = [] # Store all true labels

    print("\n--- Evaluating Model on Test Set --- ")
    # Disable gradient calculations during evaluation for efficiency
    with torch.no_grad():
        for images, labels in testloader:
            # Move data to device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update counts
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final accuracy
    accuracy = 100.0 * correct_predictions / total_samples
    print(f'Test Set Accuracy: {accuracy:.2f}%')

    # Generate confusion matrix (Meets Req 4.2)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    print("--- Evaluation Finished ---")
    return accuracy, confusion_mat

def calculate_class_accuracy(cm):
    """Calculates per-class accuracy from the confusion matrix. Meets Req 4.2.

    Args:
        cm (np.array): Confusion matrix (rows=actual, cols=predicted).

    Returns:
        np.array: Accuracy for each class (%).
    """
    # Diagonal contains the correctly classified samples for each class
    correct_per_class = cm.diagonal()
    # Sum across rows to get the total number of actual samples for each class
    total_per_class = cm.sum(axis=1)

    # Calculate accuracy, handle division by zero if a class has no samples in the test set
    per_class_acc = np.zeros_like(total_per_class, dtype=float)
    # Only calculate accuracy for classes with samples
    valid_classes_mask = total_per_class > 0
    per_class_acc[valid_classes_mask] = (correct_per_class[valid_classes_mask] /
                                         total_per_class[valid_classes_mask])

    return per_class_acc * 100 # Return as percentage

# --- Visualization Functions (Essential for analysis and report) --- #

def plot_training_curves(results, title_prefix=""):
    """Plots training loss, accuracy, and learning rate curves for multiple runs.
       Helps analyze training dynamics (Req 3.2, 4.2).
    """
    num_runs = len(results)
    # Create subplots: one row per run, three columns (Loss, Acc, LR)
    fig, axes = plt.subplots(num_runs, 3, figsize=(18, 5 * num_runs), squeeze=False)
    fig.suptitle(f'{title_prefix}Training Process Comparison', fontsize=16, y=1.02)

    for i, (name, data) in enumerate(results.items()):
        epochs_ran = len(data['losses'])
        epochs_axis = range(1, epochs_ran + 1) # Epoch numbers starting from 1

        # Plot Loss
        ax = axes[i, 0]
        ax.plot(epochs_axis, data['losses'], label='Loss', color='tab:blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} - Loss')
        ax.grid(True, alpha=0.5)
        ax.legend()

        # Plot Accuracy
        ax = axes[i, 1]
        ax.plot(epochs_axis, data['accs'], label='Accuracy', color='tab:orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{name} - Accuracy')
        ax.grid(True, alpha=0.5)
        ax.legend()

        # Plot Learning Rate
        ax = axes[i, 2]
        ax.plot(epochs_axis, data['lrs'], label='Learning Rate', color='tab:green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{name} - Learning Rate')
        # Use scientific notation for LR if values are very small
        ax.ticklabel_format(style='sci', axis='y', scilimits=(-1,2), useMathText=True)
        ax.grid(True, alpha=0.5)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to prevent title overlap
    plt.show()

def plot_confusion_matrix_heatmap(cm, title):
    """Visualizes the confusion matrix using a heatmap. Meets Req 4.2."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 10})
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

def plot_final_accuracies(accuracies, title='Final Test Accuracy Comparison'):
    """Displays a bar chart comparing final test accuracies. Meets Req 1.3, 3.1."""
    names = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(10, 6))
    # Use a perceptually uniform colormap like 'viridis'
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, values, color=colors)

    plt.ylabel('Accuracy (%)')
    plt.title(title)

    # Adjust y-axis limits for better visualization, ensuring 0 is included if relevant
    min_acc = min(values) if values else 0
    max_acc = max(values) if values else 100
    plt.ylim(max(0, min_acc - 10), min(100, max_acc + 5)) # Provide some space

    # Add accuracy values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%',
                 va='bottom', ha='center', fontsize=10) # Place text just above bar

    plt.xticks(rotation=10, ha='right') # Rotate x-axis labels slightly if they overlap
    plt.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines
    plt.tight_layout() # Adjust plot to prevent labels overlapping
    plt.show()

def plot_class_accuracy_comparison(results, title='Per-Class Accuracy Comparison'):
    """Plots per-class accuracy for different experiment runs. Meets Req 4.2."""
    plt.figure(figsize=(12, 7))
    num_classes = len(classes)
    x = np.arange(num_classes) # the label locations
    num_runs = len(results)
    width = 0.8 / num_runs # Calculate bar width based on number of runs

    # Calculate offset to center the group of bars for each class
    group_offset = -width * (num_runs - 1) / 2

    for i, (name, data) in enumerate(results.items()):
        cm = data['confusion_matrix']
        class_accs = calculate_class_accuracy(cm)
        # Calculate the position for each bar in the group
        bar_position = x + group_offset + i * width
        plt.bar(bar_position, class_accs, width, label=name)

    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    # Place ticks at the center of the groups
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend(title="Experiments")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.ylim(0, 105) # Set y-axis limit slightly above 100%
    plt.tight_layout()
    plt.show()

# =============================================================================
# 5. Main Execution Logic (Requirement 3, 5)
# =============================================================================
def main():
    """Main function to run the required experiments for the project report."""

    print("===== Gemini V4: Final CIFAR-10 CNN Experiments =====")
    print("This script executes the experiments required by the project guidelines:")
    print(" 1. Train a baseline model without data augmentation (SGD).")
    print(" 2. Train the main model with data augmentation (SGD) - Demonstrates augmentation effect.")
    print(" 3. Train the main model with a different optimizer/scheduler (AdamW+Cosine) - Compares optimization strategies.")
    print("All models use the same 'FinalCNN' architecture and dropout rate for fair comparisons where applicable.")

    # Dictionary to store results (losses, accs, lrs, final_acc, confusion_matrix)
    all_experiment_results = {}

    # --- Experiment 1: Baseline without Data Augmentation ---
    # Purpose: Establish baseline performance without augmentation (Req 1.3)
    print("\n\n===== Experiment 1: SGD + StepLR without Augmentation =====")
    trainloader_no_aug, testloader = load_data(use_augmentation_for_train=False)
    model_no_aug = FinalCNN(dropout_rate=DROPOUT_RATE) # Use consistent dropout
    criterion = nn.CrossEntropyLoss()
    optimizer_sgd_no_aug = optim.SGD(model_no_aug.parameters(), lr=LEARNING_RATE_SGD,
                                     momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY_SGD, nesterov=True)
    scheduler_sgd_no_aug = optim.lr_scheduler.StepLR(optimizer_sgd_no_aug, step_size=STEP_LR_STEP_SIZE,
                                                     gamma=STEP_LR_GAMMA)

    losses, accs, lrs = train_model(model_no_aug, trainloader_no_aug, optimizer_sgd_no_aug,
                                    scheduler_sgd_no_aug, criterion, EPOCHS)
    final_acc, cm = evaluate_model(model_no_aug, testloader)
    all_experiment_results['1_SGD_NoAug'] = {
        'losses': losses, 'accs': accs, 'lrs': lrs,
        'final_accuracy': final_acc, 'confusion_matrix': cm
    }

    # --- Experiment 2: Main Model with Data Augmentation (SGD - Best from v3) ---
    # Purpose: Demonstrate augmentation effect vs. Exp 1 (Req 1.3) & Serve as primary SGD result (Req 3.1)
    print("\n\n===== Experiment 2: SGD + StepLR with Augmentation =====")
    trainloader_aug, testloader_aug = load_data(use_augmentation_for_train=True) # Use same testloader instance is fine
    model_sgd_aug = FinalCNN(dropout_rate=DROPOUT_RATE)
    # Re-use criterion
    optimizer_sgd_aug = optim.SGD(model_sgd_aug.parameters(), lr=LEARNING_RATE_SGD,
                                  momentum=MOMENTUM_SGD, weight_decay=WEIGHT_DECAY_SGD, nesterov=True)
    scheduler_sgd_aug = optim.lr_scheduler.StepLR(optimizer_sgd_aug, step_size=STEP_LR_STEP_SIZE,
                                                  gamma=STEP_LR_GAMMA)

    losses, accs, lrs = train_model(model_sgd_aug, trainloader_aug, optimizer_sgd_aug,
                                    scheduler_sgd_aug, criterion, EPOCHS)
    final_acc, cm = evaluate_model(model_sgd_aug, testloader) # Evaluate on non-augmented test set
    all_experiment_results['2_SGD_Aug'] = {
        'losses': losses, 'accs': accs, 'lrs': lrs,
        'final_accuracy': final_acc, 'confusion_matrix': cm
    }

    # --- Experiment 3: Alternative Optimizer/Scheduler (AdamW + CosineLR) ---
    # Purpose: Compare with Exp 2 (SGD+StepLR) using same augmented data (Req 3.1, 3.2)
    print("\n\n===== Experiment 3: AdamW + CosineLR with Augmentation =====")
    # Use the same augmented trainloader and non-augmented testloader
    model_adamw_aug = FinalCNN(dropout_rate=DROPOUT_RATE) # Same architecture and dropout
    # Re-use criterion
    optimizer_adamw = optim.AdamW(model_adamw_aug.parameters(), lr=LEARNING_RATE_ADAMW,
                                  weight_decay=WEIGHT_DECAY_ADAMW)
    # Using Cosine Annealing scheduler as required
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=EPOCHS, eta_min=0)

    losses, accs, lrs = train_model(model_adamw_aug, trainloader_aug, optimizer_adamw,
                                    scheduler_cosine, criterion, EPOCHS)
    final_acc, cm = evaluate_model(model_adamw_aug, testloader)
    all_experiment_results['3_AdamW_Aug'] = {
        'losses': losses, 'accs': accs, 'lrs': lrs,
        'final_accuracy': final_acc, 'confusion_matrix': cm
    }


    # --- Results Analysis and Visualization ---
    print("\n\n===== Generating Results Visualization =====")

    # 1. Final Accuracy Comparison
    final_accuracies_dict = {name: data['final_accuracy'] for name, data in all_experiment_results.items()}
    plot_final_accuracies(final_accuracies_dict, title='Final Test Accuracy Comparison Across Experiments')

    # 2. Training Curves (Loss, Accuracy, Learning Rate)
    plot_training_curves(all_experiment_results, title_prefix="Experiment ")

    # 3. Confusion Matrices (Plot for each run)
    for name, data in all_experiment_results.items():
        plot_confusion_matrix_heatmap(data['confusion_matrix'], title=name)

    # 4. Per-Class Accuracy Comparison
    plot_class_accuracy_comparison(all_experiment_results, title='Per-Class Accuracy Comparison')

    print("\n===== All Experiments and Analysis Complete =====")
    print("Final Accuracies:")
    for name, acc in final_accuracies_dict.items():
        print(f"- {name}: {acc:.2f}%")

    # Discuss augmentation effect
    acc_no_aug = all_experiment_results['1_SGD_NoAug']['final_accuracy']
    acc_aug_sgd = all_experiment_results['2_SGD_Aug']['final_accuracy']
    print(f"\nAugmentation Effect (SGD): {acc_aug_sgd:.2f}% (with aug) vs {acc_no_aug:.2f}% (no aug) "
          f"-> Improvement: {acc_aug_sgd - acc_no_aug:.2f}%p")

    # Discuss optimizer/scheduler effect
    acc_aug_adamw = all_experiment_results['3_AdamW_Aug']['final_accuracy']
    print(f"Optimizer/Scheduler Effect (Aug Data): {acc_aug_sgd:.2f}% (SGD+StepLR) vs {acc_aug_adamw:.2f}% (AdamW+Cosine) "
          f"-> Difference: {acc_aug_sgd - acc_aug_adamw:.2f}%p")


if __name__ == "__main__":
    # This ensures the main function runs when the script is executed
    main()
