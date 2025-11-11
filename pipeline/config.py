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
        
        # Validate paths section
        required_keys = ['tiff_file', 'xml_file', 'components_dir', 
                        'embeddings_dir', 'visualizations_dir']
        for key in required_keys:
            if not self.parser.has_option('paths', key):
                raise ValueError(f"Missing required key in [paths]: {key}")
    
    def get_paths(self) -> Dict[str, Path]:
        """
        Get all file paths as Path objects.
        
        Returns:
            Dictionary with keys: tiff_file, xml_file, components_dir,
            embeddings_dir, visualizations_dir
        """
        paths = {}
        for key in self.parser.options('paths'):
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
            Dictionary with keys: query_glyph_id, top_k, generate_visualization, thumbnail_size
        """
        defaults = {
            'query_glyph_id': '',
            'top_k': 10,
            'generate_visualization': True,
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
        
        # Get boolean parameter
        if self.parser.has_option('search', 'generate_visualization'):
            params['generate_visualization'] = self.parser.getboolean('search', 'generate_visualization')
        else:
            params['generate_visualization'] = defaults['generate_visualization']
        
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
            exists = "✓" if value.exists() else "✗"
            print(f"  {key:20s}: {exists} {value}")
        
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
