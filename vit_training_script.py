"""
Training Script for Vision Transformer Gesture Recognition Model
"""
"""
# train_vit.py - Training script

1. Complete ViT architecture
2. Training pipeline with data augmentation
3. Validation and checkpointing
4. Visualization (training curves, confusion matrix)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import the VisionTransformer model from main script
# (Or copy the model definition here)

class GestureDataset(Dataset):
    """Dataset loader for gesture images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load all images
        for gesture_id in range(7):
            gesture_dir = os.path.join(root_dir, f'gesture_{gesture_id}')
            if os.path.exists(gesture_dir):
                for img_name in os.listdir(gesture_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(gesture_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(gesture_id)
        
        print(f"Loaded {len(self.images)} images")
        print(f"Label distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for gesture classification"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 n_classes=7, embed_dim=768, depth=12, n_heads=12, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                         in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embed
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


class Trainer:
    """Training manager for ViT model"""
    def __init__(self, model, train_loader, val_loader, device, 
                 learning_rate=3e-4, num_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                                     weight_decay=0.05)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )
        
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100*correct/total:.2f}%'})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Training on {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, preds, labels = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, 'best_vit_gesture_model.pth')
                print(f"✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"Training Complete! Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        return preds, labels
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Training curves saved as 'training_curves.png'")
        plt.show()
    
    def plot_confusion_matrix(self, preds, labels):
        """Plot confusion matrix"""
        gesture_names = ['NONE', 'MOVE', 'CLICK', 'RIGHT_CLICK', 
                        'DRAG', 'SCROLL_UP', 'SCROLL_DOWN']
        
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=gesture_names,
                   yticklabels=gesture_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=gesture_names))


def main():
    """Main training function"""
    # Hyperparameters
    IMG_SIZE = 224
    PATCH_SIZE = 16
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-4
    EMBED_DIM = 384
    DEPTH = 6
    N_HEADS = 6
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                              saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = GestureDataset('gesture_dataset', transform=train_transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, 
                                              [train_size, val_size])
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = VisionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        n_classes=7,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        n_heads=N_HEADS,
        mlp_ratio=4.0,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    trainer = Trainer(model, train_loader, val_loader, device, 
                     learning_rate=LEARNING_RATE, num_epochs=NUM_EPOCHS)
    preds, labels = trainer.train()
    
    # Plot results
    trainer.plot_training_curves()
    trainer.plot_confusion_matrix(preds, labels)
    
    print("\nTraining complete! Model saved as 'best_vit_gesture_model.pth'")


if __name__ == "__main__":
    main()
