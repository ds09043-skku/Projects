# 2021313549 정성수
# Introduction to Deep Learning - Project 1
# Can be run on Google Colab => It took about 97 minutes to run on Google Colab (A100 GPU)
# If the running time is too long, reducing the number of epochs or batch size may help.
# I used 50 epochs and 128 batch size.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 1. DATA TRANSFORMS
def get_transforms(use_augmentation=False):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    # Basic transform (validation and test)
    basic = transforms.Compose([
        transforms.ToTensor(),    # Convert to tensor
        transforms.Normalize(mean, std)    # Normalize
    ])
    if not use_augmentation:
        return basic
    # Augmentation transform (training)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),    # Random crop
        transforms.RandomHorizontalFlip(),    # Random horizontal flip
        transforms.RandomRotation(15),    # Random rotation
        transforms.ColorJitter(0.2,0.2,0.2,0.1),    # Color jitter
        transforms.ToTensor(),    # Convert to tensor
        transforms.Normalize(mean, std)    # Normalize
    ])

# 2. LOAD DATA WITH TRAIN/VAL/TEST SPLIT
def load_data_with_val(batch_size=128, val_ratio=0.2):
    # Download once to get indices
    base = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None)
    N = len(base)    # number of samples
    idx = torch.randperm(N).tolist()    # random list
    n_val = int(val_ratio * N)    # number of validation samples
    val_idx, train_idx = idx[:n_val], idx[n_val:]    # split

    # Create subsets with separate transforms
    train_ds = Subset(torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=get_transforms(True)), train_idx)
    val_ds = Subset(torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=get_transforms(False)), val_idx)
    test_ds = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, train_ds, val_ds

# 3. MODEL DEFINITION
class CustomCNN(nn.Module):        # Idea from VGGNET's architecture
    def __init__(self, dropout=0.3):    # dropout 0.3
        super().__init__()
        # First convolutional block (3->64->64->pooling->16*16*64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2))
        # Second convolutional block (64->128->128->pooling->8*8*128)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2))
        # Third convolutional block (128->256->256->pooling->4*4*256)
        self.conv3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2))
        # Global average pooling (4*4*256 -> 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Classifier (FC)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256,10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x).view(x.size(0),-1)
        return self.classifier(x)

# 4. TRAIN / EVAL FUNCTIONS
def train_one_epoch(model, loader, opt, sched, criterion, device):
    model.train()    # train mode
    total_loss = total_correct = total = 0
    for X,y in loader:    # batch
        X,y = X.to(device), y.to(device)
        opt.zero_grad()    # zero grad
        logits = model(X)    # forward
        loss = criterion(logits,y)    # loss
        loss.backward()    # backward
        opt.step()    # update
        total_loss += loss.item()*X.size(0)    # loss
        preds = logits.argmax(1)    # pred
        total_correct += (preds==y).sum().item()    # correct
        total += X.size(0)    # total
    if sched: sched.step()    # step
    return total_loss/total, 100*total_correct/total

def eval_model(model, loader, device):
    model.eval()    # eval mode
    total_correct=total=0 
    all_p, all_t = [],[]
    with torch.no_grad():    # no grad
        for X,y in loader:    # batch
            X,y = X.to(device), y.to(device)
            p = model(X).argmax(1)    # pred
            all_p.append(p.cpu().numpy())    # append
            all_t.append(y.cpu().numpy())    # append
            total_correct += (p==y).sum().item()    # correct
            total += X.size(0)    # total
    cm = confusion_matrix(np.concatenate(all_t), np.concatenate(all_p))    # confusion matrix
    return 100*total_correct/total, cm

# 5. PLOT CONFUSION MATRIX
def plot_confusion_matrix(cm, title, classes):
    plt.figure(figsize=(7,6))    # figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# 6. MAIN EXPERIMENT
def main():
    classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, epochs = 128, 50    # batch size 128, epochs 50    ->   97 minutes on Google Colab (A100 GPU)
    weight_decay = 5e-4
    configs = {
      'SGD_No':      {'opt':'sgd','lr':0.01, 'sched':None},    # SGD, no scheduler
      'SGD_Step':    {'opt':'sgd','lr':0.01, 'sched':'step','step':15,'gamma':0.1},    # SGD, step scheduler
      'SGD_Cosine':  {'opt':'sgd','lr':0.01, 'sched':'cos','T':epochs},    # SGD, cosine scheduler
      'Adam_No':     {'opt':'adam','lr':0.001,'sched':None},    # Adam, no scheduler
      'Adam_Step':   {'opt':'adam','lr':0.001,'sched':'step','step':15,'gamma':0.1},    # Adam, step scheduler
      'Adam_Cosine': {'opt':'adam','lr':0.001,'sched':'cos','T':epochs}    # Adam, cosine scheduler
    }

    # Load split data
    train_loader, val_loader, test_loader, train_ds, val_ds = \
        load_data_with_val(batch_size)

    results = {}
    criterion = nn.CrossEntropyLoss()

    # Train & validate each config
    for name,cfg in configs.items():
        print(f"\n>> Experiment: {name}")
        model = CustomCNN(0.3).to(device)
        opt  = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=weight_decay) \
               if cfg['opt']=='sgd' else \
               optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=weight_decay)
        sched = None
        if cfg['sched']=='step':
            sched = optim.lr_scheduler.StepLR(opt, cfg['step'], cfg['gamma'])
        elif cfg['sched']=='cos':
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['T'], eta_min=0.)

        train_losses, train_accs = [], []
        for epoch in range(epochs):
            l,a = train_one_epoch(model, train_loader, opt, sched, criterion, device)
            train_losses.append(l); train_accs.append(a)
            print(f" Epoch {epoch+1}/{epochs} - loss {l:.4f}, acc {a:.2f}%")

        val_acc, val_cm = eval_model(model, val_loader, device)
        print(f" Validation Acc: {val_acc:.2f}%")
        results[name] = {
            'train_losses':train_losses,
            'train_accs':train_accs,
            'val_acc':val_acc,
            'val_cm':val_cm,
            'config':cfg
        }

    # Select best on validation
    best = max(results, key=lambda k: results[k]['val_acc'])
    print(f"\n>>> Selected on val: {best} ({results[best]['val_acc']:.2f}%)")

    # Retrain best on train+val then test
    combined = ConcatDataset([train_ds, val_ds])
    combined_loader = DataLoader(combined, batch_size, shuffle=True, num_workers=2, pin_memory=True)

    best_cfg = results[best]['config']
    model = CustomCNN(0.3).to(device)
    opt = optim.SGD(model.parameters(), lr=best_cfg['lr'], momentum=0.9, weight_decay=weight_decay) \
          if best_cfg['opt']=='sgd' else \
          optim.Adam(model.parameters(), lr=best_cfg['lr'], weight_decay=weight_decay)
    sched = None
    if best_cfg.get('sched')=='step':
        sched = optim.lr_scheduler.StepLR(opt, best_cfg['step'], best_cfg['gamma'])
    elif best_cfg.get('sched')=='cos':
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=best_cfg['T'], eta_min=0.)

    # retrain
    for epoch in range(epochs):
        l,a = train_one_epoch(model, combined_loader, opt, sched, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - loss {l:.4f}, acc {a:.2f}%")

    test_acc, test_cm = eval_model(model, test_loader, device)
    print(f"\nFinal Test Acc: {test_acc:.2f}%")
    plot_confusion_matrix(test_cm, f'Final Test Confusion Matrix ({best})', classes)

if __name__ == "__main__":
    main()
