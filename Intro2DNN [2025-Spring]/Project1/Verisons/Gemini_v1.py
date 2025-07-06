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
# 설정 (Configuration)
# =============================================================================
# GPU 사용 가능 여부 확인 및 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device using: {device}')

# CIFAR-10 클래스 이름
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 하이퍼파라미터
BATCH_SIZE = 128
LEARNING_RATE_SGD = 0.01
LEARNING_RATE_ADAM = 0.001
WEIGHT_DECAY = 5e-4
EPOCHS = 50 # 에포크 수를 늘려 충분한 학습 기회 제공
DROPOUT_RATE = 0.4 # 드롭아웃 비율 조정
MOMENTUM = 0.9

# =============================================================================
# 1. 데이터 전처리 및 증강 (Data Preprocessing and Augmentation)
# =============================================================================
def get_transforms(use_augmentation):
    """데이터 변환 객체를 반환합니다.

    Args:
        use_augmentation (bool): 데이터 증강 사용 여부.

    Returns:
        torchvision.transforms.Compose: PyTorch 변환 객체.
    """
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if use_augmentation:
        # 데이터 증강을 포함한 변환 (훈련용)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),           # 랜덤으로 이미지 자르기 (패딩 추가)
            transforms.RandomHorizontalFlip(p=0.5),        # 50% 확률로 좌우 반전
            transforms.RandomRotation(15),               # 랜덤 회전 (각도 범위 조절 가능)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 색상 왜곡
            transforms.ToTensor(),                         # 이미지를 Tensor로 변환
            normalize                              # 정규화
        ])
    else:
        # 기본 변환 (테스트용, 정규화만 적용)
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    return transform

def load_data(use_augmentation_for_train):
    """CIFAR-10 데이터셋을 로드하고 데이터로더를 반환합니다.

    Args:
        use_augmentation_for_train (bool): 훈련 데이터셋에 증강을 적용할지 여부.

    Returns:
        tuple: (trainloader, testloader) - 훈련 및 테스트 데이터로더.
    """
    train_transform = get_transforms(use_augmentation=use_augmentation_for_train)
    test_transform = get_transforms(use_augmentation=False) # 테스트 데이터는 증강하지 않음

    # 훈련 데이터셋 로드
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # 테스트 데이터셋 로드
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    return trainloader, testloader

# =============================================================================
# 2. CNN 모델 설계 및 구현 (CNN Model Design and Implementation)
# =============================================================================
class CustomCNN(nn.Module):
    """사용자 정의 CNN 모델.

    최소 4개의 컨볼루션 레이어와 풀링, 배치 정규화, 드롭아웃을 포함합니다.
    """
    def __init__(self, dropout_rate=DROPOUT_RATE):
        super(CustomCNN, self).__init__()

        # 컨볼루션 블록 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )

        # 컨볼루션 블록 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )

        # 컨볼루션 블록 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )

        # 컨볼루션 블록 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 4x4 -> 2x2
        )

        # 분류기 (Classifier)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024), # 입력 크기 수정: 512 * 2 * 2
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10) # CIFAR-10은 10개 클래스
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 3. 모델 훈련 및 최적화 (Model Training and Optimization)
# =============================================================================
def train_model(model, trainloader, optimizer, scheduler, criterion, epochs):
    """모델을 훈련합니다.

    Args:
        model (nn.Module): 훈련할 모델.
        trainloader (DataLoader): 훈련 데이터로더.
        optimizer: 옵티마이저.
        scheduler: 학습률 스케줄러.
        criterion: 손실 함수.
        epochs (int): 훈련 에포크 수.

    Returns:
        tuple: (train_losses, train_accs) - 에포크별 훈련 손실 및 정확도 리스트.
    """
    model.to(device)
    train_losses = []
    train_accs = []

    print(f"\n--- {optimizer.__class__.__name__} testing with optimizer ---")
    for epoch in range(epochs):
        start_time = time.time()
        model.train() # 훈련 모드 설정
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 기울기 초기화
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 기록
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 스케줄러 업데이트
        if scheduler:
            # ReduceLROnPlateau는 손실값을 인자로 받음
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(epoch_loss)
            else:
                 scheduler.step()

        end_time = time.time()
        print(f'[epoch: {epoch + 1}/{epochs}] loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.2f}%, '
              f'learning rate: {optimizer.param_groups[0]["lr"]:.6f}, run time: {end_time - start_time:.2f}초')

    return train_losses, train_accs

# =============================================================================
# 4. 정량적 평가 및 결과 분석 (Quantitative Evaluation and Result Analysis)
# =============================================================================
def evaluate_model(model, testloader):
    """Evaluation.

    Args:
        model (nn.Module): Model to evaluate.
        testloader (DataLoader): Test dataloader.

    Returns:
        tuple: (accuracy, confusion_mat) - Test accuracy and confusion matrix.
    """
    model.eval() # 평가 모드 설정
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # 기울기 계산 비활성화
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'\ntest set accuracy: {accuracy:.2f}%')

    # 혼동 행렬 계산
    confusion_mat = confusion_matrix(all_labels, all_preds)
    return accuracy, confusion_mat

def plot_confusion_matrix(cm, title):
    """cisualize"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('predicted class')
    plt.ylabel('real class')
    plt.title(f'confusion matrix - {title}')
    plt.show()

def plot_training_results(results, title):
    """visualize"""
    plt.figure(figsize=(12, 5))

    # 손실 곡선
    plt.subplot(1, 2, 1)
    for name, data in results.items():
        plt.plot(data['losses'], label=f'{name} loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'{title} - test loss')
    plt.legend()
    plt.grid(True)

    # 정확도 곡선
    plt.subplot(1, 2, 2)
    for name, data in results.items():
        plt.plot(data['accs'], label=f'{name} accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy (%)')
    plt.title(f'{title} - test accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def compare_final_accuracies(accuracies):
    """visualize."""
    names = list(accuracies.keys())
    values = list(accuracies.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=plt.cm.viridis(np.linspace(0, 1, len(names))))
    plt.ylabel('Final test accuracy (%)')
    plt.title('comparison of accuracy per optimizer')
    plt.ylim(0, 100)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}%', va='bottom', ha='center') # vertically align text

    plt.show()

# =============================================================================
# 메인 실행 로직 (Main Execution Logic)
# =============================================================================
def main():
    print("=== 1. Data Augmentation ===")

    # 데이터 로드 (증강 없음 vs 증강 있음)
    trainloader_no_aug, testloader = load_data(use_augmentation_for_train=False)
    trainloader_aug, _ = load_data(use_augmentation_for_train=True) # 테스트 로더는 동일하므로 재사용

    criterion = nn.CrossEntropyLoss()

    results_augmentation = {}
    final_accuracies = {}

    # --- 데이터 증강 없이 훈련 (SGD 사용) ---
    print("\n--- without Augmentation (SGD) ---")
    model_no_aug = CustomCNN().to(device)
    optimizer_sgd_no_aug = optim.SGD(model_no_aug.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # 간단한 StepLR 스케줄러 사용 예시 (필요에 따라 다른 스케줄러 사용)
    scheduler_sgd_no_aug = optim.lr_scheduler.StepLR(optimizer_sgd_no_aug, step_size=15, gamma=0.1)

    losses, accs = train_model(model_no_aug, trainloader_no_aug, optimizer_sgd_no_aug, scheduler_sgd_no_aug, criterion, EPOCHS)
    results_augmentation['wo aug (SGD)'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_no_aug, testloader)
    final_accuracies['wo aug (SGD)'] = accuracy
    plot_confusion_matrix(cm, title='wo aug (SGD)')

    # --- 데이터 증강 적용하여 훈련 (SGD 사용) ---
    print("\n--- with Augmentation (SGD) ---")
    model_aug_sgd = CustomCNN().to(device)
    optimizer_sgd_aug = optim.SGD(model_aug_sgd.parameters(), lr=LEARNING_RATE_SGD, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_sgd_aug = optim.lr_scheduler.StepLR(optimizer_sgd_aug, step_size=15, gamma=0.1)

    losses, accs = train_model(model_aug_sgd, trainloader_aug, optimizer_sgd_aug, scheduler_sgd_aug, criterion, EPOCHS)
    results_augmentation['with aug (SGD)'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_aug_sgd, testloader)
    final_accuracies['with aug (SGD)'] = accuracy
    plot_confusion_matrix(cm, title='with aug (SGD)')

    # 데이터 증강 효과 시각화
    plot_training_results(results_augmentation, title='effect of data augmentation')

    print("\n=== 2. Comparing optimizers and scheldulars (with data augmentation) ===")

    results_optimizer = {}
    # 이전 SGD 결과 재사용
    results_optimizer['SGD + StepLR'] = results_augmentation['with aug (SGD)']

    # --- Adam 옵티마이저 + ReduceLROnPlateau 스케줄러 ---
    print("\n--- Adam + ReduceLROnPlateau test ---")
    model_adam = CustomCNN().to(device)
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY)
    # 손실이 5 에포크 동안 개선되지 않으면 학습률 0.5배 감소
    scheduler_adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=5, verbose=True)

    losses, accs = train_model(model_adam, trainloader_aug, optimizer_adam, scheduler_adam, criterion, EPOCHS)
    results_optimizer['Adam + ReduceLROnPlateau'] = {'losses': losses, 'accs': accs}
    accuracy, cm = evaluate_model(model_adam, testloader)
    final_accuracies['Adam + ReduceLR'] = accuracy
    plot_confusion_matrix(cm, title='Adam + ReduceLROnPlateau')

    # 옵티마이저/스케줄러 결과 시각화
    plot_training_results(results_optimizer, title='optimizer and schedular comparison')

    # 최종 정확도 비교 시각화
    print("\n=== 3. Final Comparison ===")
    compare_final_accuracies(final_accuracies)

if __name__ == "__main__":
    # total_start_time = time.time()
    main()
    # total_end_time = time.time()
    # print(f"\ntotal run time: {(total_end_time - total_start_time) / 60:.2f} min")
