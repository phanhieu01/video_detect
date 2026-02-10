"""Model training module for custom watermark detection models"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class WatermarkDataset(Dataset):
    """Dataset for watermark detection training"""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[Any] = None,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must match")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = cv2.imread(str(self.image_paths[idx]))
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise ValueError(f"Failed to load image or mask at index {idx}")

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)


class WatermarkDetectionModel(nn.Module):
    """Full model for watermark detection"""

    def __init__(self, input_channels: int = 3):
        super(WatermarkDetectionModel, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(input_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder
        self.dec1 = self._upconv_block(512, 256)
        self.dec2 = self._upconv_block(256, 128)
        self.dec3 = self._upconv_block(128, 64)

        # Output
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = F.max_pool2d(x1, 2)

        x3 = self.enc2(x2)
        x4 = F.max_pool2d(x3, 2)

        x5 = self.enc3(x4)
        x6 = F.max_pool2d(x5, 2)

        x7 = self.enc4(x6)
        x8 = F.max_pool2d(x7, 2)

        # Decoder
        x = self.dec1(x8)
        x = self.dec2(x)
        x = self.dec3(x)

        # Output
        x = torch.sigmoid(self.final(x))

        return x


class ModelTrainer:

    def __init__(self, config: dict = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")

        self.config = config or {}

        # Training parameters
        self.epochs = self.config.get("epochs", 50)
        self.batch_size = self.config.get("batch_size", 8)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model = WatermarkDetectionModel().to(self.device)

        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')

    def train(
        self,
        train_images: List[Path],
        train_masks: List[Path],
        val_images: Optional[List[Path]] = None,
        val_masks: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Train the model"""
        # Create datasets
        train_dataset = WatermarkDataset(train_images, train_masks)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if val_images and val_masks:
            val_dataset = WatermarkDataset(val_images, val_masks)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training loop
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                history["val_loss"].append(val_loss)

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self._save_checkpoint("best_model.pt")

            logger.info(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {train_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")

        return history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.best_loss,
        }

        save_path = Path(self.config.get("output_dir", "./models")) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["loss"]

        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def export_model(self, output_path: Path) -> None:
        """Export model for inference"""
        torch.save(self.model.state_dict(), output_path)
        logger.info(f"Exported model: {output_path}")


def prepare_training_data(
    images_dir: Path,
    annotations_file: Path,
    output_dir: Path,
    train_split: float = 0.8,
) -> Dict[str, Any]:
    """
    Prepare training data from annotated images

    Args:
        images_dir: Directory containing images
        annotations_file: JSON file with watermark annotations
        output_dir: Directory to save processed data
        train_split: Train/validation split ratio

    Returns:
        Dictionary with train/val image and mask paths
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)

    images = []
    masks = []

    for ann in annotations:
        image_path = images_dir / ann["image"]
        if not image_path.exists():
            continue

        # Create mask from annotation
        mask = np.zeros((ann["height"], ann["width"]), dtype=np.uint8)

        for region in ann.get("watermarks", []):
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            mask[y:y+h, x:x+w] = 255

        # Save mask
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        images.append(image_path)
        masks.append(mask_path)

    # Split into train/val
    num_train = int(len(images) * train_split)

    return {
        "train_images": images[:num_train],
        "train_masks": masks[:num_train],
        "val_images": images[num_train:],
        "val_masks": masks[num_train:],
    }
