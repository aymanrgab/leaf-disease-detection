import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm
from data_preprocessing import PlantVillageDataset
from model import get_model
from utils import get_random_subset, plot_training_curves, visualize_model

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_dataloaders(args):
    train_transform, val_transform = get_transforms()

    train_dataset = PlantVillageDataset('with-augmentation', transform=train_transform)
    val_dataset = PlantVillageDataset('without-augmentation', transform=val_transform)

    train_subset = get_random_subset(train_dataset, args.subset_size)
    val_subset = get_random_subset(val_dataset, args.subset_size // 5)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, len(train_dataset.classes)

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast(enabled=device.type=='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})

    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    scaler = GradScaler()
    scheduler = OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs, steps_per_epoch=len(train_loader))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{args.epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'best_model.pth'))

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
        }, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_dataloaders(args)

    model = get_model(num_classes, backbone=args.backbone, img_size=args.img_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, args)

    # Ensure the directory for saving the final model exists
    final_model_dir = 'models/saved_models'
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    # Save the final model
    final_model_path = os.path.join(final_model_dir, 'final_model.pth')
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved successfully at {final_model_path}.")

    # Visualize model predictions
    visualize_model(trained_model, val_loader, device, train_loader.dataset.dataset.classes, num_images=6)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Leaf Disease Detection Model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to use for training (cpu or cuda)')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--subset_size', type=int, default=10000, help='Size of the subset to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'], help='Backbone model to use')
    args = parser.parse_args()

    main(args)