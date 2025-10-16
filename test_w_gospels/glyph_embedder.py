import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from pathlib import Path

class GlyphEmbedder:
    def __init__(self, target_size=224, device=None):
        """
        Initialize the glyph embedder with ResNet50.
        
        Args:
            target_size: Size to pad glyphs to (224 for ResNet50)
            device: torch device ('cuda', 'cpu', or None for auto)
        """
        self.target_size = target_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained ResNet50
        print(f"Loading ResNet50 on device: {self.device}")
        self.model = models.resnet50(pretrained=True)
        
        # Remove the final classification layer to get embeddings
        # ResNet50 final pooling layer outputs 2048-dimensional features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)
        
        # ImageNet normalization (ResNet50 expects this)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def pad_to_square(self, glyph_image):
        """
        Pad glyph to square while preserving aspect ratio.
        
        Args:
            glyph_image: Grayscale glyph image (numpy array)
            
        Returns:
            Padded and resized image (224x224)
        """
        h, w = glyph_image.shape
        
        # If already square, just resize
        if h == w:
            return cv2.resize(glyph_image, (self.target_size, self.target_size))
        
        # Determine padding needed
        max_dim = max(h, w)
        
        # Create square canvas with white background (255)
        square_img = np.full((max_dim, max_dim), 255, dtype=glyph_image.dtype)
        
        # Center the glyph in the square
        if h > w:
            # Tall glyph - pad horizontally
            pad_left = (max_dim - w) // 2
            square_img[:, pad_left:pad_left + w] = glyph_image
        else:
            # Wide glyph - pad vertically  
            pad_top = (max_dim - h) // 2
            square_img[pad_top:pad_top + h, :] = glyph_image
        
        # Resize to target size
        return cv2.resize(square_img, (self.target_size, self.target_size))
    
    def preprocess_glyph(self, glyph_path):
        """
        Load and preprocess a single glyph image.
        
        Args:
            glyph_path: Path to glyph image file
            
        Returns:
            Preprocessed tensor ready for ResNet50
        """
        # Load grayscale glyph
        glyph = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
        if glyph is None:
            raise ValueError(f"Could not load glyph: {glyph_path}")
        
        # Pad to square
        padded = self.pad_to_square(glyph)
        
        # Convert to RGB (ResNet50 expects 3 channels)
        rgb_glyph = cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image for torchvision transforms
        pil_image = Image.fromarray(rgb_glyph)
        
        # Apply transforms (convert to tensor, normalize)
        tensor = self.transform(pil_image)
        
        return tensor
    
    def extract_single_embedding(self, glyph_path):
        """Extract embedding for a single glyph."""
        tensor = self.preprocess_glyph(glyph_path).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor)
            # Remove batch and spatial dimensions: (1, 2048, 1, 1) -> (2048,)
            embedding = embedding.squeeze().cpu().numpy()
        
        return embedding
    
    def extract_embeddings(self, components_dir, metadata_csv_path, batch_size=32):
        """
        Extract embeddings for all glyphs.
        
        Args:
            components_dir: Path to directory containing glyph images
            metadata_csv_path: Path to glyph metadata CSV
            batch_size: Number of glyphs to process at once
            
        Returns:
            Dictionary with glyph_ids as keys and embeddings as values
        """
        # Load metadata
        metadata = pd.read_csv(metadata_csv_path)
        print(f"Found {len(metadata)} glyphs in metadata")
        
        glyph_dir = Path(components_dir) / "glyphs"
        if not glyph_dir.exists():
            raise ValueError(f"Glyphs directory not found: {glyph_dir}")
        
        embeddings = {}
        glyph_paths = []
        glyph_ids = []
        
        # Prepare batch processing
        for _, row in metadata.iterrows():
            glyph_id = row['glyph_id']
            filename = row['filename']
            glyph_path = glyph_dir / filename
            
            if glyph_path.exists():
                glyph_paths.append(glyph_path)
                glyph_ids.append(glyph_id)
            else:
                print(f"Warning: Glyph file not found: {glyph_path}")
        
        print(f"Processing {len(glyph_paths)} glyphs in batches of {batch_size}")
        
        # Process in batches
        for i in range(0, len(glyph_paths), batch_size):
            batch_paths = glyph_paths[i:i + batch_size]
            batch_ids = glyph_ids[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                try:
                    tensor = self.preprocess_glyph(path)
                    batch_tensors.append(tensor)
                    valid_indices.append(j)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            if not batch_tensors:
                continue
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
                # Remove spatial dimensions: (B, 2048, 1, 1) -> (B, 2048)
                batch_embeddings = batch_embeddings.squeeze().cpu().numpy()
            
            # Store embeddings
            for j, embedding in enumerate(batch_embeddings):
                original_idx = valid_indices[j]
                glyph_id = batch_ids[original_idx]
                embeddings[glyph_id] = embedding
            
            print(f"Processed batch {i//batch_size + 1}/{(len(glyph_paths) + batch_size - 1)//batch_size}")
        
        print(f"Successfully extracted embeddings for {len(embeddings)} glyphs")
        return embeddings
    
    def save_embeddings(self, embeddings, metadata_csv_path, output_path):
        """
        Save embeddings along with metadata.
        
        Args:
            embeddings: Dict of glyph_id -> embedding
            metadata_csv_path: Original metadata CSV path
            output_path: Path to save embeddings and enhanced metadata
        """
        # Load original metadata
        metadata = pd.read_csv(metadata_csv_path)
        
        # Add embedding info to metadata
        metadata['has_embedding'] = metadata['glyph_id'].isin(embeddings.keys())
        
        # Save enhanced metadata
        enhanced_csv = Path(output_path) / "glyph_metadata_with_embeddings.csv"
        metadata.to_csv(enhanced_csv, index=False)
        
        # Save embeddings as numpy arrays
        embeddings_file = Path(output_path) / "glyph_embeddings.npz"
        
        # Convert to arrays for saving
        glyph_ids = list(embeddings.keys())
        embedding_matrix = np.stack([embeddings[gid] for gid in glyph_ids])
        
        np.savez_compressed(embeddings_file, 
                           embeddings=embedding_matrix,
                           glyph_ids=glyph_ids)
        
        print(f"Saved embeddings to: {embeddings_file}")
        print(f"Saved enhanced metadata to: {enhanced_csv}")
        print(f"Embedding shape: {embedding_matrix.shape}")
        
        return enhanced_csv, embeddings_file


def main():
    """Example usage"""
    
    # Configuration
    COMPONENTS_DIR = "components"
    METADATA_CSV = "components/glyph_metadata.csv"
    OUTPUT_DIR = "embeddings_output"
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create embedder
    embedder = GlyphEmbedder(target_size=224)
    
    # Extract embeddings (start with small batch for testing)
    embeddings = embedder.extract_embeddings(
        components_dir=COMPONENTS_DIR,
        metadata_csv_path=METADATA_CSV,
        batch_size=8
    )
    
    # Save results
    embedder.save_embeddings(
        embeddings=embeddings,
        metadata_csv_path=METADATA_CSV,
        output_path=OUTPUT_DIR
    )
    
    print("\nEmbedding extraction complete!")
    print(f"You can now use the embeddings for:")
    print("- Glyph similarity search")
    print("- Clustering similar glyphs")
    print("- Classification tasks")


if __name__ == "__main__":
    main()