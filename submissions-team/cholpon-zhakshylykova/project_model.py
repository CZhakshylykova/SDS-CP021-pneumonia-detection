# project_model_advanced.py

import os
import random
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, f1_score
)
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ---- 1. SET SEED EVERYTHING FOR REPRODUCIBILITY ----
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ---- 2. CONFIG ----
DATA_ROOT = "/Users/cholponzhakshylykova/Desktop/SDS/pytorch/chest_xray"
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 4
IMG_SIZE = 128
NUM_CLASSES = 2
LOG_DIR = "runs/chest_xray"
REPORTS_DIR = "reports"
PLOTS_DIR = "plots"

for folder in [REPORTS_DIR, PLOTS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ---- 3. ADVANCED DATA AUGMENTATION ----
class AlbumentationsTransform:
    def __init__(self, aug):
        self.aug = aug
    def __call__(self, img):
        return self.aug(image=np.array(img))['image']

train_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.OneOf([
        A.GaussianBlur(p=0.5),
        A.MotionBlur(p=0.5)
    ], p=0.2),
    A.Normalize([0.5], [0.5]),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize([0.5], [0.5]),
    ToTensorV2()
])

# ---- 4. DATASET + OVERSAMPLING ----
class OversampledDataset(Dataset):
    def __init__(self, normal_files, pneumonia_files, transform=None):
        n_normal, n_pneumonia = len(normal_files), len(pneumonia_files)
        if n_normal < n_pneumonia:
            normal_files = normal_files * (n_pneumonia // n_normal) + random.sample(normal_files, n_pneumonia % n_normal)
        elif n_pneumonia < n_normal:
            pneumonia_files = pneumonia_files * (n_normal // n_pneumonia) + random.sample(pneumonia_files, n_normal % n_pneumonia)
        self.images = normal_files + pneumonia_files
        self.labels = [0]*len(normal_files) + [1]*len(pneumonia_files)
        self.transform = transform

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def get_dataloaders(data_root, batch_size=32):
    train_normal = list((Path(data_root) / 'train' / 'NORMAL').glob('*.jpg')) + list((Path(data_root) / 'train' / 'NORMAL').glob('*.jpeg'))
    train_pneu = list((Path(data_root) / 'train' / 'PNEUMONIA').glob('*.jpg')) + list((Path(data_root) / 'train' / 'PNEUMONIA').glob('*.jpeg'))

    train_dataset = OversampledDataset(
        [str(x) for x in train_normal],
        [str(x) for x in train_pneu],
        transform=AlbumentationsTransform(train_aug)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.ImageFolder(Path(data_root) / 'val',
                                      transform=AlbumentationsTransform(val_aug))
    test_dataset = datasets.ImageFolder(Path(data_root) / 'test',
                                       transform=AlbumentationsTransform(val_aug))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# ---- 5. MODEL & FINE-TUNING ----
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

# ---- 6. TRAINING LOOP w/ EARLY STOPPING, SCHEDULER, MIXED PRECISION, METRICS ----
def plot_metrics(train_hist, val_hist, name):
    plt.figure(figsize=(7,5))
    plt.plot(train_hist, label='Train')
    plt.plot(val_hist, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    plt.title(f'{name} curve')
    plt.savefig(f"{PLOTS_DIR}/{name.lower()}_curve.png")
    plt.close()

def train_model(model, train_loader, val_loader, device, epochs=10, patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)
    # Weighted loss for imbalance
    class_counts = [0, 0]
    for imgs, lbls in train_loader:
        for l in lbls:
            class_counts[l] += 1
    weights = torch.FloatTensor([1/c if c>0 else 1 for c in class_counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    writer = SummaryWriter(LOG_DIR)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_acc, patience_cnt = 0, 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        avg_loss = train_loss / total
        acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(acc)
        writer.add_scalar("Loss/Train", avg_loss, epoch)
        writer.add_scalar("Acc/Train", acc, epoch)

        # ---- VALIDATION ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        all_val_preds, all_val_labels, all_val_probs = [], [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs[:,1].cpu().numpy())
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)

        val_f1 = f1_score(all_val_labels, all_val_preds)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)
        writer.add_scalar("AUC/Val", val_auc, epoch)
        writer.add_scalar("F1/Val", val_f1, epoch)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Acc={acc:.4f} | Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")

        # Learning rate scheduler
        scheduler.step(val_acc)

        # ---- EARLY STOPPING ----
        if val_acc > best_acc:
            best_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping triggered.")
                break

    plot_metrics(train_losses, val_losses, "Loss")
    plot_metrics(train_accs, val_accs, "Accuracy")
    writer.close()
    print("Best val accuracy: %.4f" % best_acc)
    return model

# ---- 7. EVALUATION (Validation/Test set): ROC, AUC, F1, confusion matrix ----
def plot_roc_curve(y_true, y_probs, filename):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def eval_model(model, data_loader, device, split="Test"):
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
    acc = test_correct / test_total
    f1 = f1_score(all_labels, all_preds)
    auc_val = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA'])
    plot_roc_curve(all_labels, all_probs, f"{PLOTS_DIR}/roc_{split.lower()}.png")
    print(f"{split} accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}")
    print(f"{split} Confusion matrix:\n{cm}")
    print(f"{split} Classification report:\n{cr}")
    # Save report
    with open(f"{REPORTS_DIR}/{split.lower()}_report.txt", "w") as f:
        f.write(f"{split} accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_val:.4f}\n")
        f.write(f"{split} Confusion matrix:\n{cm}\n")
        f.write(f"{split} Classification report:\n{cr}\n")
    return acc, f1, auc_val

# ---- 8. OPTIONAL: Grad-CAM Visualizations for explainability ----
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    grad_cam_available = True
except ImportError:
    grad_cam_available = False

def gradcam_visualization(model, data_loader, device, out_dir):
    if not grad_cam_available:
        print("Grad-CAM is not installed. Skipping CAM visualizations.")
        return
    model.eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=(device.type=='cuda'))
    os.makedirs(out_dir, exist_ok=True)
    for batch_idx, (imgs, labels) in enumerate(data_loader):
        imgs = imgs.to(device)
        grayscale_cam = cam(input_tensor=imgs, targets=None)
        for i in range(imgs.shape[0]):
            img = imgs[i].detach().cpu().numpy().transpose(1,2,0)
            img_norm = (img - img.min()) / (img.max() - img.min())
            cam_img = show_cam_on_image(img_norm, grayscale_cam[i], use_rgb=True)
            plt.imsave(f"{out_dir}/cam_{batch_idx}_{i}.png", cam_img)
        if batch_idx > 1:  # Only process a couple batches for demo
            break

# ---- 9. MAIN ----
if __name__ == "__main__":
    print("I am alive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)
    model = get_model()
    model = train_model(model, train_loader, val_loader, device, epochs=EPOCHS, patience=PATIENCE)
    # Load best model before test
    model.load_state_dict(torch.load("best_model.pth"))
    print("\n--- Validation Set Performance ---")
    eval_model(model, val_loader, device, split="Validation")
    print("\n--- Test Set Performance ---")
    eval_model(model, test_loader, device, split="Test")
    # Grad-CAM visualizations
    if grad_cam_available:
        gradcam_visualization(model, test_loader, device, out_dir=f"{PLOTS_DIR}/gradcam")
    print("\nAll metrics, plots, and reports are saved in 'plots/' and 'reports/' folders. Check TensorBoard logs in 'runs/chest_xray'.")

