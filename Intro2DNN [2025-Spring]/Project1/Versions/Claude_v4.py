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
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 디바이스: {device}')

# CIFAR-10 클래스명
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 하이퍼파라미터
BATCH_SIZE = 128
EPOCHS = 70
BASE_LR = 0.01
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

# =============================================================================
# 1. 데이터 전처리 및 증강
# =============================================================================

def get_transforms(use_augmentation=False):
    # 기본 변환 - 정규화만 적용
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 통계값
    ])
    
    if not use_augmentation:
        return base_transform
    
    # 데이터 증강 변환
    augmentation_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return augmentation_transform

# 데이터 로드
def load_data(use_augmentation=False):
    transform = get_transforms(use_augmentation)
    
    # 훈련 데이터셋
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    
    # 테스트 데이터셋 (증강 없음)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, 
        transform=get_transforms(use_augmentation=False)
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    
    return trainloader, testloader

# =============================================================================
# 2. CNN 모델 설계 및 구현
# =============================================================================

# 채널 어텐션 모듈
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

# 레지듀얼 블록 구현
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=False):
        super(ResidualBlock, self).__init__()
        
        # 첫 번째 컨볼루션 레이어
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 컨볼루션 레이어
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 어텐션 메커니즘
        self.use_attention = use_attention
        if use_attention:
            self.attention = ChannelAttention(out_channels)
        
        # 단축 연결(shortcut connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
            
        out += self.shortcut(x)  # 레지듀얼 연결
        out = F.relu(out)
        return out

# CNN 모델 설계
class CustomCNN(nn.Module):
    def __init__(self, use_attention=False, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
        # 초기 컨볼루션 레이어
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 레지듀얼 블록 레이어
        self.layer1 = self._make_layer(64, 64, 2, stride=1, use_attention=use_attention)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, use_attention=use_attention)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, use_attention=use_attention)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, use_attention=use_attention)
        
        # 분류기
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)
        self.dropout = nn.Dropout(dropout_rate)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_attention):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_attention))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_attention))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

# =============================================================================
# 3. 모델 훈련 및 최적화
# =============================================================================

def train_model(model, trainloader, optimizer, scheduler, criterion, epochs, experiment_name=""):
    model.to(device)
    
    train_losses = []
    train_accs = []
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 학습률 스케줄러 단계
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        end_time = time.time()
        
        # 매 5 에폭마다 결과 출력
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'[{experiment_name}] [에폭: {epoch + 1}/{epochs}] 손실: {epoch_loss:.4f}, 정확도: {epoch_acc:.2f}%, '
                  f'학습률: {optimizer.param_groups[0]["lr"]:.6f}, 시간: {end_time - start_time:.2f}초')
    
    return train_losses, train_accs

# =============================================================================
# 4. 모델 평가 및 결과 분석
# =============================================================================

def evaluate_model(model, testloader, experiment_name=""):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f'{experiment_name} 테스트 정확도: {accuracy:.2f}%')
    
    # 클래스별 정확도 출력
    print("\n클래스별 정확도:")
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{classes[i]}: {class_acc:.2f}%')
    
    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm, [100 * class_correct[i] / class_total[i] for i in range(10)]

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(cm, title="혼동 행렬"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('예측 클래스')
    plt.ylabel('실제 클래스')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# 모델 성능 비교 함수
def compare_results(accuracies, model_names, title="모델 정확도 비교"):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=plt.cm.viridis(np.linspace(0, 1, len(accuracies))))
    plt.ylabel('정확도 (%)')
    plt.title(title)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.5, f"{v:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# 훈련 과정 시각화 함수
def plot_training_curves(results_dict, title="훈련 과정"):
    plt.figure(figsize=(15, 6))
    
    # 손실 곡선
    plt.subplot(1, 2, 1)
    for name, data in results_dict.items():
        plt.plot(data['losses'], label=name)
    plt.xlabel('에폭')
    plt.ylabel('손실')
    plt.legend()
    plt.title('훈련 손실')
    plt.grid(True, alpha=0.3)
    
    # 정확도 곡선
    plt.subplot(1, 2, 2)
    for name, data in results_dict.items():
        plt.plot(data['accs'], label=name)
    plt.xlabel('에폭')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.title('훈련 정확도')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# 클래스별 정확도 시각화 함수
def plot_class_accuracies(class_accuracies_list, model_names, title="클래스별 정확도 비교"):
    plt.figure(figsize=(12, 8))
    
    for i, (name, accuracies) in enumerate(zip(model_names, class_accuracies_list)):
        plt.plot(range(10), accuracies, marker='o', label=name)
    
    plt.xticks(range(10), classes)
    plt.xlabel('클래스')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()

# =============================================================================
# 5. 실험 실행 및 결과 분석
# =============================================================================

def main():
    print("\n=== CIFAR-10 이미지 분류를 위한 CNN 모델 구현 ===")
    
    # 결과 저장 변수
    all_results = {}
    final_accuracies = {}
    all_confusion_matrices = {}
    all_class_accuracies = {}
    
    # ==========================================================================
    # 실험 1: 기본 모델 (SGD, 데이터 증강 없음)
    # ==========================================================================
    print("\n=== 실험 1: 기본 모델 (SGD, 데이터 증강 없음) ===")
    
    # 증강 없는 데이터 로드
    trainloader_base, testloader = load_data(use_augmentation=False)
    
    # 모델 초기화
    base_model = CustomCNN(use_attention=False).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer_sgd = optim.SGD(base_model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_sgd = CosineAnnealingLR(optimizer_sgd, T_max=EPOCHS, eta_min=0)
    
    # 모델 훈련
    experiment_name = "기본 모델 (증강 없음)"
    base_losses, base_accs = train_model(
        base_model, trainloader_base, optimizer_sgd, scheduler_sgd, criterion, EPOCHS, experiment_name
    )
    
    # 모델 평가
    base_accuracy, base_cm, base_class_accs = evaluate_model(base_model, testloader, experiment_name)
    
    # 결과 저장
    all_results[experiment_name] = {'losses': base_losses, 'accs': base_accs}
    final_accuracies[experiment_name] = base_accuracy
    all_confusion_matrices[experiment_name] = base_cm
    all_class_accuracies[experiment_name] = base_class_accs
    
    # ==========================================================================
    # 실험 2: 데이터 증강 모델 (SGD, 데이터 증강 적용)
    # ==========================================================================
    print("\n=== 실험 2: 데이터 증강 모델 (SGD, 데이터 증강 적용) ===")
    
    # 증강된 데이터 로드
    trainloader_aug, testloader = load_data(use_augmentation=True)
    
    # 모델 초기화
    aug_model = CustomCNN(use_attention=False).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    optimizer_sgd_aug = optim.SGD(aug_model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler_sgd_aug = CosineAnnealingLR(optimizer_sgd_aug, T_max=EPOCHS, eta_min=0)
    
    # 모델 훈련
    experiment_name = "데이터 증강 모델"
    aug_losses, aug_accs = train_model(
        aug_model, trainloader_aug, optimizer_sgd_aug, scheduler_sgd_aug, criterion, EPOCHS, experiment_name
    )
    
    # 모델 평가
    aug_accuracy, aug_cm, aug_class_accs = evaluate_model(aug_model, testloader, experiment_name)
    
    # 결과 저장
    all_results[experiment_name] = {'losses': aug_losses, 'accs': aug_accs}
    final_accuracies[experiment_name] = aug_accuracy
    all_confusion_matrices[experiment_name] = aug_cm
    all_class_accuracies[experiment_name] = aug_class_accs
    
    # ==========================================================================
    # 실험 3: 어텐션 모델 (AdamW, 데이터 증강 적용)
    # ==========================================================================
    print("\n=== 실험 3: 어텐션 모델 (AdamW, 데이터 증강 적용) ===")
    
    # 모델 초기화 (어텐션 메커니즘 추가)
    att_model = CustomCNN(use_attention=True).to(device)
    
    # 손실 함수 및 옵티마이저 설정 (AdamW 사용)
    optimizer_adamw = optim.AdamW(att_model.parameters(), lr=BASE_LR/10, weight_decay=WEIGHT_DECAY)
    scheduler_adamw = CosineAnnealingLR(optimizer_adamw, T_max=EPOCHS, eta_min=0)
    
    # 모델 훈련
    experiment_name = "어텐션 모델 (AdamW)"
    att_losses, att_accs = train_model(
        att_model, trainloader_aug, optimizer_adamw, scheduler_adamw, criterion, EPOCHS, experiment_name
    )
    
    # 모델 평가
    att_accuracy, att_cm, att_class_accs = evaluate_model(att_model, testloader, experiment_name)
    
    # 결과 저장
    all_results[experiment_name] = {'losses': att_losses, 'accs': att_accs}
    final_accuracies[experiment_name] = att_accuracy
    all_confusion_matrices[experiment_name] = att_cm
    all_class_accuracies[experiment_name] = att_class_accs
    
    # ==========================================================================
    # 결과 비교 및 시각화
    # ==========================================================================
    print("\n=== 결과 비교 및 시각화 ===")
    
    # 최종 정확도 비교
    compare_results(
        [final_accuracies[key] for key in final_accuracies.keys()],
        list(final_accuracies.keys()),
        title="모델 정확도 비교"
    )
    
    # 훈련 과정 시각화
    plot_training_curves(all_results, title="모델 훈련 과정 비교")
    
    # 최고 성능 모델의 혼동 행렬 시각화
    best_model = max(final_accuracies.items(), key=lambda x: x[1])[0]
    print(f"\n최고 성능 모델: {best_model}, 정확도: {final_accuracies[best_model]:.2f}%")
    plot_confusion_matrix(all_confusion_matrices[best_model], title=f"{best_model} 혼동 행렬")
    
    # 클래스별 정확도 비교
    plot_class_accuracies(
        [all_class_accuracies[key] for key in all_class_accuracies.keys()],
        list(all_class_accuracies.keys()),
        title="클래스별 정확도 비교"
    )
    
    # 결과 요약
    print("\n=== 모든 실험 완료 ===")
    for name, acc in final_accuracies.items():
        print(f"{name} 최종 정확도: {acc:.2f}%")
    
    # 데이터 증강 효과 분석
    print(f"\n데이터 증강 효과: {final_accuracies['데이터 증강 모델'] - final_accuracies['기본 모델 (증강 없음)']:.2f}%p")
    
    # 옵티마이저 비교
    print(f"옵티마이저 효과 (AdamW vs SGD): {final_accuracies['어텐션 모델 (AdamW)'] - final_accuracies['데이터 증강 모델']:.2f}%p")

if __name__ == "__main__":
    main() 
