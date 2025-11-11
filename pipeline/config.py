"""
Configuration management for manuscript processing pipeline.

This module provides centralized configuration for all three pipeline scripts
using INI-style config files. Users can create a config file specifying:
- Input paths (TIFF image, segmentation XML)
- Output directories (glyphs, embeddings, visualizations)
- Processing parameters (thresholds, batch sizes, etc.)

Example config file:
    [paths]
    tiff_file = /path/to/manuscript.tiff
    xml_file = /path/to/segmentation.xml
    components_dir = /path/to/output/components
    embeddings_dir = /path/to/output/embeddings
    visualizations_dir = /path/to/output/visualizations
    
    [extraction]
    min_area = 20
    max_area = 5000
    padding = 5
    
    [embeddings]
    target_size = 224
    batch_size = 8
"""

import configparser
from pathlib import Path
from typing import Dict, Any


class PipelineConfig:
    """Manages configuration for the manuscript processing pipeline."""
    
    def __init__(self, config_path: str):
        """
        Load configuration from INI file.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            configparser.Error: If config file is malformed
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.parser = configparser.ConfigParser()
        self.parser.read(self.config_path)
        
        # Validate required sections
        self._validate_config()
    
    def _validate_config(self):
        """Validate that all required sections and keys exist."""
        required_sections = ['paths']
        for section in required_sections:
            if not self.parser.has_section(section):
                raise ValueError(f"Missing required section: [{section}]")
        
        # Validate paths section - either tiff_file OR iiif_url must be present
        has_tiff = self.parser.has_option('paths', 'tiff_file')
        has_iiif = self.parser.has_option('paths', 'iiif_url') or self.parser.has_option('paths', 'iiif_manifest')
        
        if not (has_tiff or has_iiif):
            raise ValueError("Config must specify either 'tiff_file' or 'iiif_url'/'iiif_manifest' in [paths]")
        
        # Other required keys
        required_keys = ['xml_file', 'components_dir', 'embeddings_dir', 'visualizations_dir']
        for key in required_keys:
            if not self.parser.has_option('paths', key):
                raise ValueError(f"Missing required key in [paths]: {key}")
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Get all file paths as Path objects.
        Handles IIIF downloads if iiif_url/iiif_manifest is specified.
        
        Returns:
            Dictionary with keys: tiff_file, xml_file, components_dir,
            embeddings_dir, visualizations_dir
        """
        paths = {}
        
        # Check if we need to download from IIIF
        if self.parser.has_option('paths', 'iiif_url') or self.parser.has_option('paths', 'iiif_manifest'):
            paths = self._handle_iiif_download()
        else:
            # Standard local file paths
            for key in self.parser.options('paths'):
                value = self.parser.get('paths', key)
                paths[key] = Path(value).expanduser()
        
        return paths
    
    def _handle_iiif_download(self) -> Dict[str, Path]:
        """
        Download images from IIIF and return paths.
        
        Returns:
            Dictionary with paths including downloaded tiff_file
        """
        from iiif_downloader import IIIFDownloader
        
        # Get IIIF URL (support both 'iiif_url' and 'iiif_manifest' keys)
        iiif_url = (self.parser.get('paths', 'iiif_url', fallback=None) or 
                   self.parser.get('paths', 'iiif_manifest', fallback=None))
        
        # Get cache directory
        cache_dir = self.parser.get('paths', 'cache_dir', fallback='../cache')
        cache_dir = Path(cache_dir).expanduser()
        
        # Check if images are already downloaded
        existing_images = list(cache_dir.glob("*.tiff")) + list(cache_dir.glob("*.tif"))
        
        if existing_images:
            # Images already cached, skip download
            print(f"✓ Using {len(existing_images)} cached images from: {cache_dir}")
            tiff_file = sorted(existing_images)[0]  # Use first image as default
        else:
            # Download images
            downloader = IIIFDownloader(cache_dir=str(cache_dir))
            
            if downloader.is_manifest_url(iiif_url):
                print(f"Downloading manuscript from IIIF manifest...")
                downloaded = downloader.download_all_images(
                    manifest_url=iiif_url,
                    output_dir=str(cache_dir),
                    format='TIFF'
                )
                
                if not downloaded:
                    raise ValueError("No images downloaded from IIIF manifest")
                
                tiff_file = downloaded[0]
            else:
                print(f"Downloading single image from IIIF URL...")
                tiff_file = downloader.download_single_image(
                    image_url=iiif_url,
                    output_path=str(cache_dir / 'manuscript.tiff'),
                    format='TIFF'
                )
        
        # Build paths dictionary
        paths = {
            'tiff_file': tiff_file,
            'iiif_url': iiif_url,
            'cache_dir': cache_dir
        }
        
        # Add other paths from config
        for key in self.parser.options('paths'):
            if key not in ['iiif_url', 'iiif_manifest', 'cache_dir']:
                value = self.parser.get('paths', key)
                paths[key] = Path(value).expanduser()
        
        return paths
    
    def get_extraction_params(self) -> Dict[str, Any]:
        """
        Get glyph extraction parameters with defaults.
        
        Returns:
            Dictionary with keys: min_area, max_area, padding
        """
        defaults = {
            'min_area': 20,
            'max_area': 5000,
            'padding': 5
        }
        
        if not self.parser.has_section('extraction'):
            return defaults
        
        params = {}
        for key, default in defaults.items():
            if self.parser.has_option('extraction', key):
                params[key] = self.parser.getint('extraction', key)
            else:
                params[key] = default
        
        return params
    
    def get_embeddings_params(self) -> Dict[str, Any]:
        """
        Get embedding generation parameters with defaults.
        
        Returns:
            Dictionary with keys: model, target_size, batch_size
        """
        defaults = {
            'model': 'hog',  # Options: hog, resnet50, trocr
            'target_size': 64,  # 64 for HOG, 224 for ResNet/TrOCR
            'batch_size': 8
        }
        
        if not self.parser.has_section('embeddings'):
            return defaults
        
        params = {}
        
        # Get model type (string)
        if self.parser.has_option('embeddings', 'model'):
            params['model'] = self.parser.get('embeddings', 'model').strip().lower()
        else:
            params['model'] = defaults['model']
        
        # Get numeric parameters
        for key in ['target_size', 'batch_size']:
            if self.parser.has_option('embeddings', key):
                params[key] = self.parser.getint('embeddings', key)
            else:
                params[key] = defaults[key]
        
        return params
    
    def get_search_params(self) -> Dict[str, Any]:
        """
        Get similarity search parameters with defaults.
        
        Returns:
            Dictionary with keys: query_glyph_id, top_k, generate_visualization, 
                                  generate_html_report, thumbnail_size
        """
        defaults = {
            'query_glyph_id': '',
            'top_k': 10,
            'generate_visualization': True,
            'generate_html_report': True,
            'thumbnail_size': 100
        }
        
        if not self.parser.has_section('search'):
            return defaults
        
        params = {}
        
        # Get query_glyph_id (string)
        if self.parser.has_option('search', 'query_glyph_id'):
            params['query_glyph_id'] = self.parser.get('search', 'query_glyph_id').strip()
        else:
            params['query_glyph_id'] = defaults['query_glyph_id']
        
        # Get numeric parameters
        for key in ['top_k', 'thumbnail_size']:
            if self.parser.has_option('search', key):
                params[key] = self.parser.getint('search', key)
            else:
                params[key] = defaults[key]
        
        # Get boolean parameters
        for key in ['generate_visualization', 'generate_html_report']:
            if self.parser.has_option('search', key):
                params[key] = self.parser.getboolean('search', key)
            else:
                params[key] = defaults[key]
        
        return params
    
    def get_iiif_params(self) -> Dict[str, Any]:
        """
        Get IIIF download parameters with defaults.
        
        Returns:
            Dictionary with keys: max_images, delay, format
        """
        defaults = {
            'max_images': None,  # Download all images
            'delay': 0.5,  # Delay between requests in seconds
            'format': 'TIFF'  # Output format
        }
        
        if not self.parser.has_section('iiif'):
            return defaults
        
        params = {}
        
        # Get max_images (optional)
        if self.parser.has_option('iiif', 'max_images'):
            params['max_images'] = self.parser.getint('iiif', 'max_images')
        else:
            params['max_images'] = defaults['max_images']
        
        # Get delay
        if self.parser.has_option('iiif', 'delay'):
            params['delay'] = self.parser.getfloat('iiif', 'delay')
        else:
            params['delay'] = defaults['delay']
        
        # Get format
        if self.parser.has_option('iiif', 'format'):
            params['format'] = self.parser.get('iiif', 'format').strip().upper()
        else:
            params['format'] = defaults['format']
        
        return params
    
    def ensure_directories_exist(self):
        """Create all output directories if they don't exist."""
        paths = self.get_paths()
        
        # Create components directory
        components_glyphs = paths['components_dir'] / 'glyphs'
        components_glyphs.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings directory
        paths['embeddings_dir'].mkdir(parents=True, exist_ok=True)
        
        # Create visualizations directory
        paths['visualizations_dir'].mkdir(parents=True, exist_ok=True)
    
    def print_config(self):
        """Print the loaded configuration to console."""
        print("\n" + "="*60)
        print("Pipeline Configuration")
        print("="*60)
        
        paths = self.get_paths()
        print("\n[Paths]")
        for key, value in paths.items():
            # Handle both Path objects and strings
            if isinstance(value, Path):
                exists = "✓" if value.exists() else "✗"
                print(f"  {key:20s}: {exists} {value}")
            else:
                # For strings (like iiif_url), just print the value
                print(f"  {key:20s}: {value}")
        
        extraction = self.get_extraction_params()
        print("\n[Extraction Parameters]")
        for key, value in extraction.items():
            print(f"  {key:20s}: {value}")
        
        embeddings = self.get_embeddings_params()
        print("\n[Embedding Parameters]")
        for key, value in embeddings.items():
            print(f"  {key:20s}: {value}")
        
        search = self.get_search_params()
        print("\n[Search Parameters]")
        for key, value in search.items():
            print(f"  {key:20s}: {value}")
        
        print("="*60 + "\n")
