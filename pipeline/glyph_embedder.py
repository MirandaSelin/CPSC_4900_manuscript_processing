import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage import exposure
import os
import argparse
from pathlib import Path
from config import PipelineConfig

class GlyphEmbedder:
    def __init__(self, model_type='hog', target_size=64):
        """
        Initialize the glyph embedder with specified model.
        
        Supported models:
        - hog: Histogram of Oriented Gradients (traditional CV, fast, good for characters)
        - trocr: TrOCR transformer (deep learning for handwriting, slower but more powerful)
        - resnet50: ResNet50 CNN (pretrained on ImageNet, may not work well for text)
        
        Args:
            model_type: Type of embedding model ('hog', 'trocr', or 'resnet50')
            target_size: Size to resize glyphs to
        """
        self.model_type = model_type.lower()
        self.target_size = target_size
        
        # Initialize the appropriate model
        if self.model_type == 'hog':
            self._init_hog()
        elif self.model_type == 'trocr':
            self._init_trocr()
        elif self.model_type == 'resnet50':
            self._init_resnet50()
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'hog', 'trocr', or 'resnet50'")
    
    def _init_hog(self):
        """Initialize HOG (Histogram of Oriented Gradients) feature extractor"""
        # HOG parameters optimized for character recognition
        self.orientations = 9  # Number of orientation bins (9 is standard)
        self.pixels_per_cell = (8, 8)  # Size of cells for gradient computation
        self.cells_per_block = (2, 2)  # Normalization blocks
        
        print(f"Initialized HOG embedder:")
        print(f"  Target size: {self.target_size}x{self.target_size}")
        print(f"  Orientations: {self.orientations}")
        print(f"  Pixels per cell: {self.pixels_per_cell}")
        print(f"  Cells per block: {self.cells_per_block}")
    
    def _init_trocr(self):
        """Initialize TrOCR (Transformer for OCR) feature extractor"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError:
            raise ImportError(
                "TrOCR requires transformers and torch. Install with:\n"
                "  pip install transformers torch pillow"
            )
        
        print("Loading TrOCR model (this may take a moment)...")
        
        # Use the handwritten model
        model_name = "microsoft/trocr-base-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Set to evaluation mode
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Initialized TrOCR embedder:")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Target size: {self.target_size}x{self.target_size}")
    
    def _init_resnet50(self):
        """Initialize ResNet50 feature extractor"""
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
        except ImportError:
            raise ImportError(
                "ResNet50 requires torch and torchvision. Install with:\n"
                "  pip install torch torchvision"
            )
        
        print("Loading ResNet50 model...")
        
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Remove final classification layer to get embeddings
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.target_size, self.target_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized ResNet50 embedder:")
        print(f"  Device: {self.device}")
        print(f"  Target size: {self.target_size}x{self.target_size}")
    
    def preprocess_glyph(self, glyph_image):
        """
        Preprocess glyph based on the selected model.
        
        Args:
            glyph_image: Grayscale glyph image (numpy array)
            
        Returns:
            Preprocessed image ready for the model
        """
        if self.model_type == 'hog':
            return self._preprocess_hog(glyph_image)
        elif self.model_type == 'trocr':
            return self._preprocess_trocr(glyph_image)
        elif self.model_type == 'resnet50':
            return self._preprocess_resnet50(glyph_image)
    
    def _preprocess_hog(self, glyph_image):
        """Preprocess glyph for HOG feature extraction"""
        h, w = glyph_image.shape
        
        # Make square by padding
        max_dim = max(h, w)
        square_img = np.full((max_dim, max_dim), 255, dtype=glyph_image.dtype)
        
        # Center the glyph
        if h > w:
            pad_left = (max_dim - w) // 2
            square_img[:, pad_left:pad_left + w] = glyph_image
        else:
            pad_top = (max_dim - h) // 2
            square_img[pad_top:pad_top + h, :] = glyph_image
        
        # Resize to target size
        resized = cv2.resize(square_img, (self.target_size, self.target_size))
        
        # Invert so text is white on black (HOG works better this way)
        inverted = cv2.bitwise_not(resized)
        
        return inverted
    
    def _preprocess_trocr(self, glyph_image):
        """Preprocess glyph for TrOCR (uses PIL Image)"""
        from PIL import Image
        
        # TrOCR expects RGB PIL images, so we need to convert
        h, w = glyph_image.shape
        
        # Make square by padding
        max_dim = max(h, w)
        square_img = np.full((max_dim, max_dim), 255, dtype=glyph_image.dtype)
        
        # Center the glyph
        if h > w:
            pad_left = (max_dim - w) // 2
            square_img[:, pad_left:pad_left + w] = glyph_image
        else:
            pad_top = (max_dim - h) // 2
            square_img[pad_top:pad_top + h, :] = glyph_image
        
        # Resize to target size
        resized = cv2.resize(square_img, (self.target_size, self.target_size))
        
        # Convert to RGB PIL Image
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        return pil_image
    
    def _preprocess_resnet50(self, glyph_image):
        """Preprocess glyph for ResNet50 (just returns grayscale for transform)"""
        h, w = glyph_image.shape
        
        # Make square by padding
        max_dim = max(h, w)
        square_img = np.full((max_dim, max_dim), 255, dtype=glyph_image.dtype)
        
        # Center the glyph
        if h > w:
            pad_left = (max_dim - w) // 2
            square_img[:, pad_left:pad_left + w] = glyph_image
        else:
            pad_top = (max_dim - h) // 2
            square_img[pad_top:pad_top + h, :] = glyph_image
        
        return square_img
    
    
    def extract_single_embedding(self, glyph_path):
        """
        Extract embedding for a single glyph using the configured model.
        
        Args:
            glyph_path: Path to glyph image file
            
        Returns:
            Feature vector (1D numpy array)
        """
        # Load grayscale glyph
        glyph = cv2.imread(str(glyph_path), cv2.IMREAD_GRAYSCALE)
        if glyph is None:
            raise ValueError(f"Could not load glyph: {glyph_path}")
        
        # Preprocess based on model
        preprocessed = self.preprocess_glyph(glyph)
        
        # Extract features based on model
        if self.model_type == 'hog':
            return self._extract_hog_features(preprocessed)
        elif self.model_type == 'trocr':
            return self._extract_trocr_features(preprocessed)
        elif self.model_type == 'resnet50':
            return self._extract_resnet50_features(preprocessed)
    
    def _extract_hog_features(self, preprocessed_glyph):
        """Extract HOG features from preprocessed glyph"""
        features = hog(
            preprocessed_glyph,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            feature_vector=True
        )
        return features
    
    def _extract_trocr_features(self, preprocessed_glyph):
        """Extract TrOCR encoder features from preprocessed glyph (PIL Image)"""
        import torch
        
        # Process image with TrOCR processor
        pixel_values = self.processor(preprocessed_glyph, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Extract encoder features (not decoder output)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(pixel_values)
            # Use the [CLS] token embedding (first token) as the glyph representation
            # Shape: [batch_size, seq_len, hidden_size] -> we take [0, 0, :] for first image, first token
            features = encoder_outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return features
    
    def _extract_resnet50_features(self, preprocessed_glyph):
        """Extract ResNet50 features from preprocessed glyph"""
        import torch
        
        # Apply transforms
        img_tensor = self.transform(preprocessed_glyph).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features
    
    def extract_embeddings(self, components_dir, metadata_csv_path, batch_size=32):
        """
        Extract embeddings for all glyphs using the configured model.
        
        Args:
            components_dir: Path to directory containing glyph images
            metadata_csv_path: Path to glyph metadata CSV
            batch_size: Batch size for processing (used by deep learning models)
            
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
        
        # Process each glyph
        for idx, row in metadata.iterrows():
            glyph_id = row['glyph_id']
            filename = row['filename']
            glyph_path = glyph_dir / filename
            
            if not glyph_path.exists():
                print(f"Warning: Glyph file not found: {glyph_path}")
                continue
            
            try:
                # Extract features using the configured model
                features = self.extract_single_embedding(glyph_path)
                embeddings[glyph_id] = features
                
                # Progress update every 100 glyphs
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1}/{len(metadata)} glyphs")
                    
            except Exception as e:
                print(f"Error processing {glyph_path}: {e}")
        
        print(f"Successfully extracted {self.model_type.upper()} embeddings for {len(embeddings)} glyphs")
        
        # Print feature vector size
        if embeddings:
            sample_embedding = next(iter(embeddings.values()))
            print(f"{self.model_type.upper()} feature dimension: {len(sample_embedding)}")
        
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
    """Process glyphs and generate embeddings using configured model"""
    
    parser = argparse.ArgumentParser(
        description='Generate embeddings for extracted glyphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python glyph_embedder.py gospel_config.ini
  python glyph_embedder.py /path/to/my_manuscript_config.ini

Supported Models:
  hog      - Histogram of Oriented Gradients (traditional CV, fast, good for characters)
  trocr    - TrOCR transformer (deep learning for handwriting, slower but powerful)
  resnet50 - ResNet50 CNN (pretrained on ImageNet, may not work well for text)

Note:
  The model type is specified in the config file under [embeddings] section.
  TrOCR and ResNet50 require additional dependencies (transformers, torch).
        '''
    )
    
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='gospel_config.ini',
        help='Path to configuration file (default: gospel_config.ini)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = PipelineConfig(args.config)
        config.print_config()
        config.ensure_directories_exist()
        
        # Get paths and parameters
        paths = config.get_paths()
        embeddings_params = config.get_embeddings_params()
        
        # Create embedder with configured model and parameters
        print(f"\nUsing model: {embeddings_params['model']}")
        embedder = GlyphEmbedder(
            model_type=embeddings_params['model'],
            target_size=embeddings_params['target_size']
        )
        
        # Prepare paths
        metadata_csv = paths['components_dir'] / 'glyph_metadata.csv'
        
        # Extract embeddings
        print("\nStarting embedding generation...")
        embeddings = embedder.extract_embeddings(
            components_dir=str(paths['components_dir']),
            metadata_csv_path=str(metadata_csv),
            batch_size=embeddings_params['batch_size']
        )
        
        # Save results
        embedder.save_embeddings(
            embeddings=embeddings,
            metadata_csv_path=str(metadata_csv),
            output_path=str(paths['embeddings_dir'])
        )
        
        print("\n✓ Embedding extraction complete!")
        print(f"Model used: {embeddings_params['model']}")
        print(f"You can now use the embeddings for:")
        print("  - Glyph similarity search")
        print("  - Clustering similar glyphs")
        print("  - Classification tasks")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except ImportError as e:
        print(f"Import error: {e}")
        print(f"\nMake sure you have the required dependencies installed.")
        print(f"For TrOCR: pip install transformers torch pillow")
        print(f"For ResNet50: pip install torch torchvision")


if __name__ == "__main__":
    main()