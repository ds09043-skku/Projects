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

# =============================================================================
# Configuration
# =============================================================================
# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE_SGD = 0.01
LEARNING_RATE_ADAM = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 50 # Number of epochs for training
# Increased dropout rate based on results analysis to potentially reduce overfitting
DROPOUT_RATE = 0.5
MOMENTUM = 0.9 # Momentum for SGD optimizer
# Adjusted ReduceLROnPlateau patience based on results analysis
SCHEDULER_PATIENCE = 7
SCHEDULER_FACTOR = 0.5

# =============================================================================
# 1. Data Preprocessing and Augmentation
# =============================================================================
def get_transforms(use_augmentation):
    """Returns data transformation pipelines.

    Args:
        use_augmentation (bool): Whether to apply data augmentation.

    Returns:
        torchvision.transforms.Compose: A PyTorch transforms pipeline.
    """
    # Normalization parameters for CIFAR-10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if use_augmentation:
        # Transformation pipeline with data augmentation for training
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),           # Randomly crop the image with padding
            transforms.RandomHorizontalFlip(p=0.5),        # Randomly flip the image horizontally
            transforms.RandomRotation(15),               # Randomly rotate the image
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Randomly change color properties
            transforms.ToTensor(),                         # Convert image to PyTorch Tensor
            normalize                              # Normalize the tensor
        ])
    else:
        # Basic transformation pipeline for testing (ToTensor + Normalize)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    return transform

def load_data(use_augmentation_for_train):
    """Loads the CIFAR-10 dataset and returns DataLoader objects.

    Args:
        use_augmentation_for_train (bool): Apply augmentation to the training set.

    Returns:
        tuple: (trainloader, testloader) - DataLoaders for training and testing.
    """
    train_transform = get_transforms(use_augmentation=use_augmentation_for_train)
    test_transform = get_transforms(use_augmentation=False) # No augmentation for the test set

    # Load training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    # Create DataLoader for training set
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # Load test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    # Create DataLoader for test set
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return trainloader, testloader

# =============================================================================
# 2. CNN Model Design and Implementation
# =============================================================================
class CustomCNN(nn.Module):
    """Custom Convolutional Neural Network model.

    Contains four convolutional blocks, each with Conv2D, BatchNorm2d, ReLU,
    followed by MaxPool2d. Includes a classifier with Dropout for regularization.
    """
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super(CustomCNN, self).__init__()

        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # Input: 3x32x32, Output: 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Output: 64x32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 64x16x16
        )

        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # Output: 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # Output: 128x16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 128x8x8
        )

        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # Output: 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # Output: 256x8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 256x4x4
        )

        # Convolutional Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # Output: 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # Output: 512x4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: 512x2x2
        )

        # Classifier (Fully Connected Layers)
        self.classifier = nn.Sequential(
            nn.Flatten(), # Flatten the feature map: 512 * 2 * 2 = 2048
            nn.Linear(512 * 2 * 2, 1024), # Input features: 2048, Output features: 1024
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), # Apply dropout for regularization
            nn.Linear(1024, 512), # Input features: 1024, Output features: 512
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), # Apply dropout again
            nn.Linear(512, 10) # Output features: 10 (number of CIFAR-10 classes)
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 3. Model Training and Optimization
# =============================================================================
def train_model(model, trainloader, optimizer, scheduler, criterion, epochs):
    """Trains the model for a specified number of epochs.

    Args:
        model (nn.Module): The neural network model to train.
        trainloader (DataLoader): DataLoader for the training data.
        optimizer: The optimization algorithm (e.g., SGD, Adam).
        scheduler: Learning rate scheduler.
        criterion: The loss function.
        epochs (int): The number of epochs to train for.

    Returns:
        tuple: (train_losses, train_accs) - Lists of training loss and accuracy per epoch.

    Note:
        Early Stopping based on validation set performance is highly recommended
        to prevent overfitting, especially when training for a fixed number of epochs.
        It involves monitoring validation loss/accuracy and stopping training
        when performance on the validation set stops improving.
    """
    model.to(device) # Move model to the configured device (GPU or CPU)
    train_losses = []
    train_accs = []

    print(f"\n--- Starting Training with {optimizer.__class__.__name__} --- LrScheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    for epoch in range(epochs):
        start_time = time.time()
        model.train() # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / total_samples
        epoch_acc = 100.0 * correct_predictions / total_samples
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Store current learning rate before scheduler step
        current_lr = optimizer.param_groups[0]['lr']

        # Step the learning rate scheduler
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(epoch_loss) # Step based on training loss
            else:
                 scheduler.step()

        end_time = time.time()
        print(f'[Epoch: {epoch + 1}/{epochs}] Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, '
              f'LR: {current_lr:.6f}, Time: {end_time - start_time:.2f}s')

    return train_losses, train_accs

# =============================================================================
# 4. Quantitative Evaluation and Result Analysis
# =============================================================================
def evaluate_model(model, testloader):
    """Evaluates the model on the test dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        testloader (DataLoader): DataLoader for the test data.

    Returns:
        tuple: (accuracy, confusion_mat) - Test accuracy and confusion matrix.
    """
    model.eval() # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations during evaluation
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct_predictions / total_samples
    print(f'\nTest Set Accuracy: {accuracy:.2f}%')
    confusion_mat = confusion_matrix(all_labels, all_preds)
    return accuracy, confusion_mat

def plot_confusion_matrix(cm, title):
    """Visualizes the confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

def plot_training_results(results, title):
    """Visualizes training loss and accuracy curves for different runs."""
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['losses'], label=f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Training Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['accs'], label=f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_final_accuracies(accuracies):
    """Displays a bar chart comparing final test accuracies of different runs."""
    names = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Comparison of Final Accuracies by Optimizer/Settings')
    plt.ylim(max(0, min(values) - 5), min(100, max(values) + 5)) # Adjust y-lim based on values

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center')

    plt.show()

# =============================================================================
# Main Execution Logic
# =============================================================================
def main():
    """Main function to run the experiments based on project guidelines."""
    print("=== Experiment 1: Effect of Data Augmentation (using SGD) ===")

    trainloader_no_aug, testloader = load_data(use_augmentation_for_train=False)
    trainloader_aug, _ = load_data(use_augmentation_for_train=True)
    criterion = nn.CrossEntropyLoss()

    results_augmentation = {}
    final_accuracies = {}

    # --- Train without Data Augmentation (SGD + StepLR for baseline comparison) ---
    print("\n--- Training without Augmentation (SGD + StepLR) ---")
    model_no_aug = CustomCNN().to(device)
    # Use the original dropout rate for this baseline model
    model_no_aug.classifier[-2] = nn.Dropout(0.4)
    model_no_aug.classifier[-5] = nn.Dropout(0.4)

    optimizer_sgd_no_aug = optim.SGD(model_no_aug.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # Use StepLR as in the original notebook execution for direct comparison
    scheduler_sgd_no_aug = optim.lr_scheduler.StepLR(optimizer_sgd_no_aug, step_size=15, gamma=0.1)

    losses, accs = train_model(model_no_aug, trainloader_no_aug, optimizer_sgd_no_aug, scheduler_sgd_no_aug, criterion, EPOCHS)
    results_augmentation['No Aug (SGD+StepLR)'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_no_aug, testloader)
    final_accuracies['No Aug (SGD+StepLR)'] = accuracy
    # plot_confusion_matrix(cm, title='No Augmentation (SGD + StepLR)') # Optionally plot

    # --- Train with Data Augmentation (SGD + StepLR for baseline comparison) ---
    print("\n--- Training with Augmentation (SGD + StepLR) ---")
    model_aug_sgd = CustomCNN().to(device)
    # Use the original dropout rate for this baseline model
    model_aug_sgd.classifier[-2] = nn.Dropout(0.4)
    model_aug_sgd.classifier[-5] = nn.Dropout(0.4)

    optimizer_sgd_aug = optim.SGD(model_aug_sgd.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_sgd_aug = optim.lr_scheduler.StepLR(optimizer_sgd_aug, step_size=15, gamma=0.1)

    losses, accs = train_model(model_aug_sgd, trainloader_aug, optimizer_sgd_aug, scheduler_sgd_aug, criterion, EPOCHS)
    results_augmentation['With Aug (SGD+StepLR)'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_aug_sgd, testloader)
    final_accuracies['With Aug (SGD+StepLR)'] = accuracy
    # plot_confusion_matrix(cm, title='With Augmentation (SGD + StepLR)') # Optionally plot

    plot_training_results(results_augmentation, title='Effect of Data Augmentation (SGD+StepLR)')

    print("\n=== Experiment 2: Optimizer Comparison (Adam with improvements vs SGD Baseline) ===")

    results_optimizer = {}
    # Include the SGD baseline result for comparison
    results_optimizer['SGD + StepLR'] = results_augmentation['With Aug (SGD+StepLR)']

    # --- Train with Adam Optimizer + ReduceLROnPlateau (Improved Settings) ---
    print("\n--- Training with Adam + ReduceLROnPlateau (Improved Settings) ---")
    # Create model with the NEW dropout rate (0.5)
    model_adam_improved = CustomCNN(dropout_rate=DROPOUT_RATE).to(device)
    optimizer_adam = optim.Adam(model_adam_improved.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY)
    # Use ReduceLROnPlateau with ADJUSTED patience
    scheduler_adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE, verbose=True)

    losses, accs = train_model(model_adam_improved, trainloader_aug, optimizer_adam, scheduler_adam, criterion, EPOCHS)
    results_optimizer['Adam + ReduceLR (Improved)'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_adam_improved, testloader)
    final_accuracies['Adam + ReduceLR (Improved)'] = accuracy
    plot_confusion_matrix(cm, title='Adam + ReduceLROnPlateau (Improved Settings)')

    # Visualize the comparison including the improved Adam run
    plot_training_results(results_optimizer, title='Optimizer Comparison (Augmented Data)')

    # --- Final Accuracy Comparison --- #
    print("\n=== Final Accuracy Comparison Across All Runs ===")
    compare_final_accuracies(final_accuracies)

    print("\n=== All Experiments Complete ===")

if __name__ == "__main__":
    main()
