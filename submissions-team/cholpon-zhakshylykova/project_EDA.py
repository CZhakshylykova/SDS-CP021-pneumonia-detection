"""
Chest X-ray Pneumonia Detection: Exploratory Data Analysis & Data Preprocessing
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Author: Cholpon Zhakshylykova
"""

import os
import random
import warnings
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

# Configuration
warnings.filterwarnings("ignore")
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create folders if not exist
os.makedirs("plots", exist_ok=True)

# Redirect all print output to reports.txt
sys.stdout = open("reports.txt", "w")

class ChestXrayEDA:
    """Comprehensive EDA class for Chest X-ray Pneumonia dataset"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.splits = ['train', 'val', 'test']
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.dataset_stats = {}
        self._validate_dataset_structure()
    
    def _validate_dataset_structure(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.data_root}")
        for split in self.splits:
            split_path = self.data_root / split
            if not split_path.exists():
                raise FileNotFoundError(f"Split directory not found: {split_path}")
            for cls in self.classes:
                class_path = split_path / cls
                if not class_path.exists():
                    raise FileNotFoundError(f"Class directory not found: {class_path}")
    
    def analyze_dataset_distribution(self) -> Dict:
        stats = {}
        for split in self.splits:
            stats[split] = {}
            split_path = self.data_root / split
            for cls in self.classes:
                class_path = split_path / cls
                image_files = [f for f in class_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
                stats[split][cls] = len(image_files)
            stats[split]['total'] = sum(stats[split].values())
        self.dataset_stats = stats

        # Print statistics (these go to reports.txt)
        print("="*60)
        print("DATASET DISTRIBUTION ANALYSIS")
        print("="*60)
        for split in self.splits:
            print(f"\n{split.upper()} SET:")
            for cls in self.classes:
                count = stats[split][cls]
                percentage = (count / stats[split]['total']) * 100
                print(f"  {cls:>10}: {count:>5} images ({percentage:.1f}%)")
            print(f"  {'TOTAL':>10}: {stats[split]['total']:>5} images")
        total_images = sum(stats[split]['total'] for split in self.splits)
        total_normal = sum(stats[split]['NORMAL'] for split in self.splits)
        total_pneumonia = sum(stats[split]['PNEUMONIA'] for split in self.splits)
        print(f"\nOVERALL DATASET:")
        print(f"  {'NORMAL':>10}: {total_normal:>5} images ({(total_normal/total_images)*100:.1f}%)")
        print(f"  {'PNEUMONIA':>10}: {total_pneumonia:>5} images ({(total_pneumonia/total_images)*100:.1f}%)")
        print(f"  {'TOTAL':>10}: {total_images:>5} images")
            # Assess class imbalance and recommend oversampling if needed
        imbalance_info = []
        imbalance_threshold = 1.2  # You can set this threshold as needed
        
        for split in self.splits:
            n_normal = stats[split]['NORMAL']
            n_pneumonia = stats[split]['PNEUMONIA']
            ratio = max(n_normal, n_pneumonia) / (min(n_normal, n_pneumonia) + 1e-9)
            imbalance_info.append((split, ratio))
        
        overall_ratio = max(total_normal, total_pneumonia) / (min(total_normal, total_pneumonia) + 1e-9)
        
        print("\nCLASS IMBALANCE ANALYSIS & RECOMMENDATION:")
        for split, ratio in imbalance_info:
            print(f"  {split.upper()} set imbalance ratio: {ratio:.2f} (max/min)")
        print(f"  OVERALL imbalance ratio: {overall_ratio:.2f} (max/min)")
        
        if overall_ratio > imbalance_threshold:
            print("\nRecommendation: There is a significant class imbalance.")
            print("It is recommended to use OVERSAMPLING (or class weighting) during model training to address this.")
        else:
            print("\nNo significant class imbalance detected.")
        
        return stats

    
    def visualize_distribution(self):
        if not self.dataset_stats:
            self.analyze_dataset_distribution()
        splits = list(self.dataset_stats.keys())
        normal_counts = [self.dataset_stats[split]['NORMAL'] for split in splits]
        pneumonia_counts = [self.dataset_stats[split]['PNEUMONIA'] for split in splits]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        x = np.arange(len(splits))
        width = 0.6
        axes[0, 0].bar(x, normal_counts, width, label='NORMAL', alpha=0.8)
        axes[0, 0].bar(x, pneumonia_counts, width, bottom=normal_counts, label='PNEUMONIA', alpha=0.8)
        axes[0, 0].set_xlabel('Dataset Split')
        axes[0, 0].set_ylabel('Number of Images')
        axes[0, 0].set_title('Dataset Distribution by Split')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([s.capitalize() for s in splits])
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        x = np.arange(len(splits))
        width = 0.35
        axes[0, 1].bar(x - width/2, normal_counts, width, label='NORMAL', alpha=0.8)
        axes[0, 1].bar(x + width/2, pneumonia_counts, width, label='PNEUMONIA', alpha=0.8)
        axes[0, 1].set_xlabel('Dataset Split')
        axes[0, 1].set_ylabel('Number of Images')
        axes[0, 1].set_title('Class Distribution Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([s.capitalize() for s in splits])
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        total_normal = sum(normal_counts)
        total_pneumonia = sum(pneumonia_counts)
        axes[1, 0].pie([total_normal, total_pneumonia], labels=['NORMAL', 'PNEUMONIA'],
                       autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[1, 0].set_title('Overall Class Distribution')
        imbalance_ratios = []
        split_labels = []
        for split in splits:
            normal = self.dataset_stats[split]['NORMAL']
            pneumonia = self.dataset_stats[split]['PNEUMONIA']
            ratio = pneumonia / normal if normal > 0 else 0
            imbalance_ratios.append(ratio)
            split_labels.append(f"{split.capitalize()}\n({pneumonia}:{normal})")
        bars = axes[1, 1].bar(range(len(splits)), imbalance_ratios, alpha=0.8)
        axes[1, 1].set_xlabel('Dataset Split')
        axes[1, 1].set_ylabel('Pneumonia:Normal Ratio')
        axes[1, 1].set_title('Class Imbalance by Split')
        axes[1, 1].set_xticks(range(len(splits)))
        axes[1, 1].set_xticklabels(split_labels)
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Balanced')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        for i, (bar, ratio) in enumerate(zip(bars, imbalance_ratios)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            f'{ratio:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        # --- Save the distribution plot ---
        fig.savefig(os.path.join("plots", "dataset_distribution.png"))
        plt.close(fig)
    
    def sample_images_visualization(self, n_samples: int = 8):
        fig, axes = plt.subplots(2, n_samples, figsize=(20, 8))
        for class_idx, class_name in enumerate(self.classes):
            class_path = self.data_root / 'train' / class_name
            image_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg'))
            sampled_files = random.sample(image_files, min(n_samples, len(image_files)))
            for img_idx, img_path in enumerate(sampled_files):
                try:
                    img = Image.open(img_path).convert('L')
                    axes[class_idx, img_idx].imshow(img, cmap='gray')
                    axes[class_idx, img_idx].set_title(f'{class_name}\n{img_path.name}')
                    axes[class_idx, img_idx].axis('off')
                except Exception as e:
                    axes[class_idx, img_idx].text(0.5, 0.5, f'Error loading\n{img_path.name}',
                                                 ha='center', va='center', transform=axes[class_idx, img_idx].transAxes)
                    axes[class_idx, img_idx].axis('off')
        plt.suptitle('Sample Images from Each Class', fontsize=16)
        plt.tight_layout()
        # --- Save the sample images plot ---
        fig.savefig(os.path.join("plots", "sample_images.png"))
        plt.close(fig)


def visualize_augmentations(data_root: str, n_augmentations: int = 5):
    transform_augment = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])
    sample_path = Path(data_root) / 'train' / 'PNEUMONIA'
    sample_files = list(sample_path.glob('*.jpeg')) + list(sample_path.glob('*.jpg'))
    sample_img_path = random.choice(sample_files)
    original_img = Image.open(sample_img_path).convert('L')
    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(20, 4))
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    for i in range(n_augmentations):
        augmented = transform_augment(original_img)
        axes[i + 1].imshow(augmented.squeeze(), cmap='gray')
        axes[i + 1].set_title(f'Augmented {i + 1}')
        axes[i + 1].axis('off')
    plt.suptitle(f'Data Augmentation Examples - {sample_img_path.name}', fontsize=14)
    plt.tight_layout()
    # --- Save augmentation plot ---
    fig.savefig(os.path.join("plots", "data_augmentation.png"))
    plt.close(fig)


def main():
    DATA_ROOT = "/Users/cholponzhakshylykova/Desktop/SDS/pytorch/chest_xray"  # Update this path!
    try:
        print("🫁 CHEST X-RAY PNEUMONIA DETECTION - EDA & PREPROCESSING")
        print("=" * 70)
        eda = ChestXrayEDA(DATA_ROOT)
        print("\n📊 ANALYZING DATASET DISTRIBUTION...")
        eda.analyze_dataset_distribution()
        print("\n🎨 CREATING VISUALIZATIONS...")
        eda.visualize_distribution()
        print("\n🖼️  DISPLAYING SAMPLE IMAGES...")
        eda.sample_images_visualization()
        print("\n🔄 DEMONSTRATING DATA AUGMENTATION...")
        visualize_augmentations(DATA_ROOT)
        print("\n✅ PLOTS SAVED IN 'plots/' FOLDER, REPORTS SAVED IN 'reports.txt'.")
    except FileNotFoundError as e:
        print(f"❌ ERROR: {e}")
        print("Please update the DATA_ROOT variable with the correct path to your dataset.")
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")


if __name__ == "__main__":
    main()
