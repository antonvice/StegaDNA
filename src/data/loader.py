import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import glob
from loguru import logger
from typing import List, Tuple, Optional
import hashlib

class StegaDataset(Dataset):
    """
    Dataset for StegaDNA project.
    Pairs images with technical message payloads extracted from TSV tables.
    """
    
    def __init__(
        self, 
        tsv_path: str, 
        image_dir: str, 
        image_size: int = 256, 
        payload_bits: int = 64,
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            tsv_path: Path to the TSV file containing message payloads.
            image_dir: Directory containing source images.
            image_size: Targeted image resolution (NxN).
            payload_bits: Number of bits per message.
            transform: Optional torchvision transforms.
        """
        self.image_size = image_size
        self.payload_bits = payload_bits
        
        # Load and analyze text data from the 'table'
        logger.info(f"Loading metadata from {tsv_path}")
        try:
            # We use 'message_text' column as payload source
            self.df = pd.read_csv(tsv_path, sep='\t')
            self.df = self.df[self.df['message_text'].notna()]
            self.texts = self.df['message_text'].astype(str).tolist()
            logger.info(f"Successfully extracted {len(self.texts)} text records.")
        except Exception as e:
            logger.error(f"Failed to load TSV data: {e}")
            raise

        # Load image paths from recursive search
        logger.info(f"Scanning for images in {image_dir}")
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "**/*.jpg"), recursive=True))
        logger.info(f"Found {len(self.image_paths)} images.")

        if not self.image_paths:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")

        # Default transforms if none provided
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        # Return the smaller of the two to ensure valid pairing
        return min(len(self.texts), len(self.image_paths))

    def _text_to_bits(self, text: str) -> torch.Tensor:
        """
        Converts text string to a fixed-length bit vector using Reed-Solomon encoding.
        Allows the model to correct bit flips automatically during decoding.
        """
        import reedsolo
        # For 128 bits (16 bytes), we'll do 10 bytes payload + 6 bytes ECC.
        # This allows correcting up to 3 full bytes of errors.
        ecc_bytes = 6
        payload_bytes = (self.payload_bits // 8) - ecc_bytes
        if payload_bytes < 1:
            payload_bytes = self.payload_bits // 8
            ecc_bytes = 0
            
        text_bytes = text.encode('utf-8')[:payload_bytes].ljust(payload_bytes, b'\x00')
        
        if ecc_bytes > 0:
            rs = reedsolo.RSCodec(ecc_bytes)
            encoded = rs.encode(text_bytes)
        else:
            encoded = text_bytes
            
        bits = []
        for byte in encoded:
            # We enforce 8 bits per byte cleanly
            for i in range(8):
                if len(bits) < self.payload_bits:
                    bits.append((byte >> i) & 1)
                    
        # Pad with zeros safely if somehow shorter
        while len(bits) < self.payload_bits:
            bits.append(0)
            
        return torch.tensor(bits[:self.payload_bits], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (Image Tensor, Payload Bit-Vector)
        """
        img_path = self.image_paths[idx]
        text = self.texts[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Skipping to next.")
            return self.__getitem__((idx + 1) % len(self))
            
        payload = self._text_to_bits(text)
        
        return image, payload

def get_stega_dataloaders(
    tsv_path: str, 
    image_dir: str, 
    batch_size: int = 32, 
    image_size: int = 256, 
    payload_bits: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates both Train and Validation DataLoaders.
    """
    dataset = StegaDataset(
        tsv_path=tsv_path,
        image_dir=image_dir,
        image_size=image_size,
        payload_bits=payload_bits
    )
    
    # Split indices
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split = int(num_samples * (1 - val_ratio))
    
    import random
    random.seed(42) # Reproducible split
    random.shuffle(indices)
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_stega_dataloader(
    tsv_path: str, 
    image_dir: str, 
    batch_size: int = 32, 
    image_size: int = 256, 
    payload_bits: int = 64,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Legacy helper for backward compatibility.
    """
    dataset = StegaDataset(
        tsv_path=tsv_path,
        image_dir=image_dir,
        image_size=image_size,
        payload_bits=payload_bits
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )

if __name__ == "__main__":
    # Test the loader logic
    TSV_FILE = "/Users/antonvice/Documents/programming/StegaDNA/data/text/150k_msgs_sample_hashed_pii.tsv"
    IMG_DIR = "/Users/antonvice/Documents/programming/StegaDNA/data/images"
    
    loader = get_stega_dataloader(TSV_FILE, IMG_DIR, batch_size=4)
    images, payloads = next(iter(loader))
    
    print(f"Batch images shape: {images.shape}")
    print(f"Batch payloads shape: {payloads.shape}")
    print(f"Sample payload: {payloads[0]}")
