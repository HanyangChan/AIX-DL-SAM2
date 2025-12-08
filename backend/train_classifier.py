print("Script started...")
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loaders(data_dir, batch_size=32, sample_ratio=1.0):
    """
    Creates DataLoaders for train and validation sets.
    Assumes data_dir has 'train' and 'test' (or 'val') subdirectories.
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'test') # Assuming 'test' is used for validation/eval
    
    if not os.path.exists(val_dir) and os.path.exists(os.path.join(data_dir, 'val')):
        val_dir = os.path.join(data_dir, 'val')

    # Check if train/test structure exists
    if os.path.exists(train_dir) and os.path.exists(val_dir):
        print(f"Found standard train/test structure in {data_dir}")
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    else:
        # Fallback: Flat structure (auto-split)
        print(f"Standard train/test structure not found. Attempting automatic split from {data_dir}...")
        full_dataset = datasets.ImageFolder(data_dir, transform=train_transform) # Use train transform initially
        
        # Split 80/20
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # We need to apply val_transform to val_dataset. 
        # Since random_split preserves the underlying dataset's transform, we need a wrapper or just accept train_transform for val (suboptimal but works).
        # Better approach: Copy dataset for val
        import copy
        val_dataset.dataset = copy.deepcopy(full_dataset)
        val_dataset.dataset.transform = val_transform
        
        # For class names, we can access the underlying dataset
        train_dataset.classes = full_dataset.classes

    # Subset if requested
    if sample_ratio < 1.0:
        print(f"Subsetting dataset to {sample_ratio*100}%...")
        
        def subset_dataset(dataset):
            num_samples = int(len(dataset) * sample_ratio)
            indices = torch.randperm(len(dataset))[:num_samples]
            return torch.utils.data.Subset(dataset, indices)
            
        train_dataset = subset_dataset(train_dataset)
        val_dataset = subset_dataset(val_dataset)
        
        # Restore classes attribute for Subset
        if hasattr(train_dataset.dataset, 'classes'):
             train_dataset.classes = train_dataset.dataset.classes
        elif hasattr(train_dataset.dataset, 'dataset') and hasattr(train_dataset.dataset.dataset, 'classes'):
             train_dataset.classes = train_dataset.dataset.dataset.classes

    print(f"Using {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_dataset.classes

def initialize_model(num_classes):
    """
    Initializes EfficientNet V2 Small model.
    """
    print("Initializing EfficientNet V2 Small...")
    model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
    
    # Modify the classifier head
    # EfficientNet V2 S classifier structure: Sequential(Dropout, Linear)
    # We replace the Linear layer
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(device)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_acc = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  Saved best model with Acc: {best_acc:.4f}")
            
    # Save metadata
    import json
    with open('model_metadata.json', 'w') as f:
        json.dump({'class_names': train_loader.dataset.classes}, f)
    print("Saved model_metadata.json")

    print(f'Training complete. Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model, val_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_loaded.png')
    plt.close()
    print("Saved confusion_matrix_loaded.png")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    return all_preds, all_labels

# --- Grad-CAM ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
            
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def visualize_gradcam(model, val_loader, class_names, num_images=5):
    print("Generating Grad-CAM visualizations...")
    # Target layer for EfficientNet V2 S: features[-1] (the last conv block)
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    
    images_shown = 0
    
    # Get a batch
    inputs, labels = next(iter(val_loader))
    inputs = inputs.to(device)
    
    for i in range(min(num_images, len(inputs))):
        img_tensor = inputs[i:i+1]
        label = labels[i].item()
        
        cam = grad_cam(img_tensor, class_idx=label)
        
        # Denormalize image for visualization
        img = inputs[i].cpu().permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1] # BGR to RGB
        
        cam_img = heatmap + img
        cam_img = cam_img / np.max(cam_img)
        
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original: {class_names[label]}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cam_img)
        plt.title(f"Grad-CAM")
        plt.axis('off')
        
        plt.savefig(f'gradcam_{i}.png')
        plt.close()
        images_shown += 1
        
    print(f"Saved {images_shown} Grad-CAM images.")

# --- t-SNE ---
def visualize_tsne(model, val_loader, class_names, max_samples=500):
    print("Generating t-SNE visualization...")
    model.eval()
    features = []
    labels_list = []
    
    # Hook to get features before classifier
    # For EfficientNet, we can use the output of avgpool
    # But simpler: just run forward pass up to avgpool
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            # EfficientNet V2 forward pass parts
            x = model.features(inputs)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            
            features.extend(x.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            
            if len(features) >= max_samples:
                break
                
    features = np.array(features)
    labels_list = np.array(labels_list)
    
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=[class_names[i] for i in labels_list],
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full",
        alpha=0.7
    )
    plt.title('t-SNE visualization of Image Embeddings')
    plt.savefig('tsne_loaded.png')
    plt.close()
    print("Saved tsne_loaded.png")

def main():
    parser = argparse.ArgumentParser(description='Train Food Classifier')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='Ratio of dataset to use (0.0-1.0)')
    parser.add_argument('--dry_run', action='store_true', help='Run a single batch for testing')
    
    args = parser.parse_args()
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("  WARNING: CUDA not available. Training will be slow on CPU.")
    
    # 1. Load Data
    train_loader, val_loader, class_names = get_data_loaders(args.data_dir, args.batch_size, args.sample_ratio)
    print(f"Classes: {class_names}")
    
    # 2. Initialize Model
    model = initialize_model(len(class_names))
    
    # 3. Pre-training Evaluation (Baseline)
    print("\n--- Pre-training Evaluation ---")
    evaluate_model(model, val_loader, class_names)
    
    if args.dry_run:
        print("Dry run enabled. Exiting after pre-training eval.")
        return

    # 4. Train
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs)
    
    # 5. Post-training Evaluation
    print("\n--- Post-training Evaluation ---")
    evaluate_model(model, val_loader, class_names)
    
    # 6. Advanced Visualization
    visualize_gradcam(model, val_loader, class_names)
    visualize_tsne(model, val_loader, class_names)

if __name__ == '__main__':
    main()
