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

# GPU 사용 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 중인 장치: {device}')

# CIFAR-10 클래스명
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 데이터 전처리 및 증강
def get_transforms(use_augmentation=False):
    # 기본 변환 - 정규화만 적용
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 데이터 증강 변환
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # 랜덤 수평 뒤집기
        transforms.RandomCrop(32, padding=4),    # 랜덤 자르기
        transforms.RandomRotation(10),           # 랜덤 회전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 지터
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return augmentation_transform if use_augmentation else base_transform

# 데이터 로드
def load_data(use_augmentation=False):
    transform = get_transforms(use_augmentation)
    
    # 훈련 데이터셋
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    
    # 테스트 데이터셋 (증강 없음)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, 
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    
    return trainloader, testloader

# CNN 모델 정의
class CustomCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CustomCNN, self).__init__()
        
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 두 번째 컨볼루션 블록
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 세 번째 컨볼루션 블록
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 네 번째 컨볼루션 블록
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # 풀링 레이어
        self.pool = nn.MaxPool2d(2, 2)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)
        
        # 완전 연결 레이어
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # 블록 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 블록 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 블록 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 블록 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # 피쳐맵을 1차원으로 펼침
        x = x.view(-1, 512 * 2 * 2)
        
        # 완전 연결 레이어
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 모델 훈련 함수
def train_model(model, trainloader, optimizer, scheduler=None, epochs=30):
    criterion = nn.CrossEntropyLoss()
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
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        end_time = time.time()
        print(f'[에포크: {epoch + 1}] 손실: {epoch_loss:.4f}, 정확도: {epoch_acc:.2f}%, '
              f'학습률: {optimizer.param_groups[0]["lr"]:.6f}, 소요시간: {end_time - start_time:.2f}초')
    
    return train_losses, train_accs

# 모델 평가 함수
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
    print(f'테스트 세트 정확도: {accuracy:.2f}%')
    
    # 혼동 행렬 생성
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

# 혼동 행렬 시각화 함수
def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('예측 클래스')
    plt.ylabel('실제 클래스')
    plt.title('혼동 행렬')
    plt.show()

# 두 가지 학습 방법의 성능 비교 함수
def compare_results(model1_acc, model2_acc, model1_name, model2_name):
    labels = [model1_name, model2_name]
    accuracies = [model1_acc, model2_acc]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, accuracies, color=['skyblue', 'orange'])
    plt.ylabel('정확도 (%)')
    plt.title('모델 정확도 비교')
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    plt.ylim(0, 100)
    plt.show()

def main():
    # 1. 데이터 증강 없이 모델 훈련
    print("\n=== 데이터 증강 없이 모델 훈련 ===")
    trainloader_no_aug, testloader = load_data(use_augmentation=False)
    
    model_no_aug = CustomCNN().to(device)
    
    # SGD 옵티마이저 사용
    optimizer_sgd = optim.SGD(model_no_aug.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler_sgd = optim.lr_scheduler.CosineAnnealingLR(optimizer_sgd, T_max=200)
    
    train_losses_no_aug, train_accs_no_aug = train_model(
        model_no_aug, trainloader_no_aug, optimizer_sgd, scheduler_sgd, epochs=30
    )
    
    accuracy_no_aug, cm_no_aug = evaluate_model(model_no_aug, testloader)
    
    # 2. 데이터 증강 적용하여 모델 훈련
    print("\n=== 데이터 증강 적용하여 모델 훈련 ===")
    trainloader_aug, _ = load_data(use_augmentation=True)
    
    model_aug = CustomCNN().to(device)
    
    # Adam 옵티마이저 사용
    optimizer_adam = optim.Adam(model_aug.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler_adam = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, mode='min', factor=0.5, patience=5, verbose=True)
    
    train_losses_aug, train_accs_aug = train_model(
        model_aug, trainloader_aug, optimizer_adam, scheduler_adam, epochs=30
    )
    
    accuracy_aug, cm_aug = evaluate_model(model_aug, testloader)
    
    # 3. 결과 비교 및 시각화
    print("\n=== 결과 비교 및 시각화 ===")
    
    # 정확도 비교
    compare_results(accuracy_no_aug, accuracy_aug, "증강 없음", "데이터 증강 적용")
    
    # 혼동 행렬 시각화
    print("\n데이터 증강 없음 - 혼동 행렬:")
    plot_confusion_matrix(cm_no_aug)
    
    print("\n데이터 증강 적용 - 혼동 행렬:")
    plot_confusion_matrix(cm_aug)
    
    # 학습 손실 및 정확도 곡선
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_no_aug, 'b-', label='증강 없음')
    plt.plot(train_losses_aug, 'r-', label='데이터 증강')
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.legend()
    plt.title('훈련 손실')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs_no_aug, 'b-', label='증강 없음')
    plt.plot(train_accs_aug, 'r-', label='데이터 증강')
    plt.xlabel('에포크')
    plt.ylabel('정확도 (%)')
    plt.legend()
    plt.title('훈련 정확도')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
