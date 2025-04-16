import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Class to abstract loading of the multi modal dataset
class MultiModalDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths, numeric features, and labels.
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        # Attributes to maintain and manage different parts of the dataset
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Numeric/ feature columns from the dataset
        # Note removed null columns and meaningless columns (e.g., X and Y locations in mosaic)
        self.feature_cols = [
            'Area..ABD.', 'Area..Filled.',
            'Aspect.Ratio', 'Calibration.Factor', 'Calibration.Image',
            'Circle.Fit', 'Circularity', 'Circularity..Hu.', 'Compactness',
            'Convex.Perimeter', 'Convexity', 'Diameter..ABD.', 'Diameter..ESD.',
            'Diameter..FD.', 'Edge.Gradient', 'Elongation', 'Feret.Angle.Max',
            'Feret.Angle.Min', 'Fiber.Curl', 'Fiber.Straightness', 'Filter.Score',
            'Geodesic.Aspect.Ratio', 'Geodesic.Length', 'Geodesic.Thickness',
            'Image.Height', 'Image.Width', 'Intensity', 'Length',
            'Particles.Per.Chain', 'Perimeter', 'Roughness', 'Sigma.Intensity',
            'Source.Image', 'Sphere.Complement', 'Sphere.Count', 'Sphere.Unknown',
            'Sphere.Volume', 'Sum.Intensity', 'Symmetry', 'Transparency',
            'Volume..ABD.', 'Volume..ESD.', 'Width']

        self.label_col = "int_labels" # Class label
        self.path_col = "resized_filepath" # Filepath of resized image (50x50)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Row number to be returned

        Returns:
            dict: A dictionary containing:
                - 'image' (torch.Tensor): Transformed image tensor of shape [1, H, W]
                - 'numerical' (torch.Tensor): Tensor of numeric features of shape [num_features] 
                - 'label' (torch.Tensor): Integer tensor containing the class label
        """
        row = self.data.iloc[idx] # Get relevant row from dataset

        # Load image
        img_path = os.path.join(self.image_dir, row[self.path_col])
        image = Image.open(img_path).convert("L")  # Ensure grayscale ("L" mode)

        if self.transform:
            image = self.transform(image)  # e.g. ToTensor, Normalize, etc.

        # Numeric features
        numeric_feats = torch.tensor(np.array(row[self.feature_cols].tolist(), dtype=np.float32)) # Ensure all numeric

        # Label
        label = torch.tensor(row[self.label_col], dtype=torch.long)

        return {
            'image': image,          # shape: [1, H, W] after transform
            'numerical': numeric_feats,  # shape: [num_features]
            'label': label
        }


# Build model
class MultiModalNet(nn.Module): 
    def __init__(self, num_numeric_features= 43, num_classes=9):
        """
        Args:
            num_numeric_features: Number of numeric input features (e.g., 3).
            num_classes: Number of output classes for final classification.
        """
        super(MultiModalNet, self).__init__()

        # Load the pretrained EfficientNet model
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modify EfficientNetModel for 1 channel and 50x50 images
        old_conv = self.effnet.features[0][0]  # first Conv2d layer
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride= 1, #old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )

        # Copy over/average the weights from 3 channels into 1
        with torch.no_grad():
            new_conv.weight = nn.Parameter(torch.mean(old_conv.weight, dim=1, keepdim=True))
        self.effnet.features[0][0] = new_conv
        effnet_out_dim = self.effnet.classifier[1].in_features

        # Replace the classifier with an identity operation so we can get raw features
        self.effnet.classifier = nn.Identity()

        # Build Feed Forward network for numeric data
        self.ff = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        ff_out_dim = 32

        # Combine two sub networks
        combined_dim = effnet_out_dim + ff_out_dim
        self.classifier = nn.Linear(combined_dim, num_classes)

    def forward(self, image, numeric):
        # 1) Extract features from EfficientNet
        cnn_features = self.effnet(image)  # shape: [batch_size, 1280]

        # 2) Extract features from feed-forward layers
        ff_features = self.ff(numeric)     # shape: [batch_size, 32]

        # 3) Concatenate
        combined = torch.cat([cnn_features, ff_features], dim=1)  # [batch_size, 1280 + 32]

        # 4) Classify
        logits = self.classifier(combined) # [batch_size, num_classes]
        return logits


# Training run loop
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train() # Set to train to track gradients for back prop
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        # Move batch of training data to GPU from CPU
        images = batch['image'].to(device)       # [B, 1, H, W]
        numeric = batch['numerical'].to(device)  # [B, num_numeric_features]
        labels = batch['label'].to(device)       # [B]

        optimizer.zero_grad()

        outputs = model(images, numeric)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Validation run loop
def validate_one_epoch(model, dataloader, device):
    model.eval()
    confusion_matrix = torch.zeros(9, 9)  # 9x9 matrix for 9 classes

    # We won't compute gradients during validation
    with torch.no_grad():
        for batch in dataloader:
            # Move to GPU
            images = batch['image'].to(device)
            numeric = batch['numerical'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, numeric)
            _, preds = torch.max(outputs, 1)

            # Update confusion matrix
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    return confusion_matrix


def main():
    img_size = 50
    num_classes = 9
    csv_file = r"combined_data.csv"  # CSV file with image paths, numeric features, and label
    image_dir = r"data"              # Directory with images
    batch_size = 256
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Run on GPU if possible
    print("cuda" if torch.cuda.is_available() else "cpu")


    print("Transforming and preparing Dataset")
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to 50x50 pixels
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalizing for grayscale images
    ])


    # Build the dataset
    full_dataset = MultiModalDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)

    #Split into 70% Train, 20% Validate, 10% test
    indices = np.arange(len(full_dataset))
    labels = full_dataset.data["int_labels"] # Collect labels

    # First split: 90% (train+val) and 10% test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.1, stratify=labels, random_state=42)
    # Second split: of the 90%, split into train and validate
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=2/9, stratify=labels.iloc[train_val_idx], random_state=42)

    test_dataset = Subset(full_dataset, test_idx)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = MultiModalNet(num_numeric_features=43, num_classes= num_classes).to(device)
    best_model_path = "best_multimodal_efficientnet_9class_FINAL.pth"

    model.load_state_dict(torch.load(best_model_path))
    print("Loaded Model")
    confusion_mat= validate_one_epoch(model, test_loader, device)
    # Save confusion matrix to file
    np.save('confusion_matrix_vF.npy', confusion_mat)
    print("Saved confusion matrix to confusion_matrix_vFINAL.npy")



if __name__ == "__main__":
    main()
