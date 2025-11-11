"""
IIIF Manifest Downloader

Downloads images from IIIF manifests for manuscript processing.
Supports both single image URLs and full manifest URLs.
"""

import requests
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
import argparse
from typing import List, Dict, Optional
import time
from config import PipelineConfig


class IIIFDownloader:
    def __init__(self, cache_dir: str = "../cache"):
        """
        Initialize IIIF downloader.
        
        Args:
            cache_dir: Directory to cache downloaded images
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def is_manifest_url(self, url: str) -> bool:
        """Check if URL is likely a manifest (vs direct image)"""
        # Manifests typically don't have image extensions
        # and often contain 'manifest' or 'presentation' in path
        return (
            'manifest' in url.lower() or
            '/presentation/' in url or
            (not url.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')))
        )
    
    def download_manifest(self, manifest_url: str) -> Dict:
        """
        Download and parse IIIF manifest.
        
        Args:
            manifest_url: URL to IIIF manifest JSON
            
        Returns:
            Parsed manifest dictionary
        """
        print(f"Downloading manifest from: {manifest_url}")
        response = requests.get(manifest_url)
        response.raise_for_status()
        
        manifest = response.json()
        print(f"✓ Manifest downloaded successfully")
        
        # Print basic info
        if 'label' in manifest:
            label = manifest['label']
            if isinstance(label, dict):
                # IIIF v3 format
                title = next(iter(label.values()))[0] if label else "Unknown"
            else:
                # IIIF v2 format
                title = label
            print(f"  Title: {title}")
        
        if 'items' in manifest:
            print(f"  Pages: {len(manifest['items'])}")
        elif 'sequences' in manifest:
            # IIIF v2
            canvases = manifest['sequences'][0].get('canvases', [])
            print(f"  Pages: {len(canvases)}")
            
        return manifest
    
    def extract_image_urls(self, manifest: Dict) -> List[Dict[str, str]]:
        """
        Extract all image URLs from a IIIF manifest.
        
        Args:
            manifest: Parsed IIIF manifest
            
        Returns:
            List of dicts with 'url', 'label', and 'image_id'
        """
        images = []
        
        # IIIF Presentation API v3
        if 'items' in manifest:
            for canvas in manifest['items']:
                label = self._get_label(canvas.get('label', {}))
                
                # Navigate to image URL
                if 'items' in canvas:
                    for annotation_page in canvas['items']:
                        if 'items' in annotation_page:
                            for annotation in annotation_page['items']:
                                if 'body' in annotation:
                                    body = annotation['body']
                                    if 'id' in body:
                                        image_url = body['id']
                                        # Extract image ID from URL
                                        image_id = self._extract_image_id(image_url)
                                        
                                        images.append({
                                            'url': image_url,
                                            'label': label,
                                            'image_id': image_id
                                        })
        
        # IIIF Presentation API v2 (fallback)
        elif 'sequences' in manifest:
            for canvas in manifest['sequences'][0].get('canvases', []):
                label = self._get_label(canvas.get('label', ''))
                
                if 'images' in canvas:
                    for image in canvas['images']:
                        if 'resource' in image:
                            image_url = image['resource'].get('@id', image['resource'].get('id', ''))
                            image_id = self._extract_image_id(image_url)
                            
                            images.append({
                                'url': image_url,
                                'label': label,
                                'image_id': image_id
                            })
        
        print(f"✓ Found {len(images)} images in manifest")
        return images
    
    def _get_label(self, label) -> str:
        """Extract label text from IIIF label field"""
        if isinstance(label, dict):
            # IIIF v3: {"none": ["label"]} or {"en": ["label"]}
            values = next(iter(label.values())) if label else []
            return values[0] if values else "Unknown"
        elif isinstance(label, str):
            return label
        else:
            return "Unknown"
    
    def _extract_image_id(self, url: str) -> str:
        """Extract image ID from IIIF URL"""
        # URL format: https://host/iiif/2/IMAGE_ID/full/full/0/default.jpg
        parts = url.split('/')
        for i, part in enumerate(parts):
            if part == 'iiif' and i + 2 < len(parts):
                return parts[i + 2]
        # Fallback: use last part of path before query
        return url.split('/')[-1].split('?')[0].split('.')[0]
    
    def download_image(self, image_url: str, output_path: Path, 
                      format: str = 'TIFF', delay: float = 0.5) -> Path:
        """
        Download a single image from IIIF URL.
        
        Args:
            image_url: URL to image
            output_path: Path to save image
            format: Image format (TIFF, PNG, JPEG)
            delay: Delay between requests (be polite to servers)
            
        Returns:
            Path to downloaded image
        """
        # Check cache first
        if output_path.exists():
            print(f"  ✓ Using cached: {output_path.name}")
            return output_path
        
        # Download
        time.sleep(delay)  # Be polite to servers
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Convert to PIL Image
        img = Image.open(BytesIO(response.content))
        
        # Save in requested format
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format)
        
        print(f"  ✓ Downloaded: {output_path.name} ({img.size[0]}x{img.size[1]})")
        
        return output_path
    
    def download_all_images(self, manifest_url: str, output_dir: Optional[str] = None,
                           format: str = 'TIFF', delay: float = 0.5,
                           max_images: Optional[int] = None) -> List[Path]:
        """
        Download all images from a IIIF manifest.
        
        Args:
            manifest_url: URL to IIIF manifest
            output_dir: Directory to save images (default: cache_dir)
            format: Image format (TIFF, PNG, JPEG)
            delay: Delay between downloads in seconds
            max_images: Maximum number of images to download (None = all)
            
        Returns:
            List of paths to downloaded images
        """
        # Download and parse manifest
        manifest = self.download_manifest(manifest_url)
        
        # Extract image URLs
        image_infos = self.extract_image_urls(manifest)
        
        if max_images:
            image_infos = image_infos[:max_images]
            print(f"Limiting to first {max_images} images")
        
        # Set output directory
        if output_dir is None:
            output_dir = self.cache_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download all images
        downloaded_paths = []
        print(f"\nDownloading {len(image_infos)} images...")
        
        for i, info in enumerate(image_infos, 1):
            print(f"[{i}/{len(image_infos)}] {info['label']}")
            
            # Create filename from label and image_id
            safe_label = "".join(c if c.isalnum() or c in ('-', '_') else '_' 
                                for c in info['label'])
            filename = f"{i:03d}_{safe_label}_{info['image_id']}.{format.lower()}"
            output_path = output_dir / filename
            
            try:
                path = self.download_image(info['url'], output_path, format, delay)
                downloaded_paths.append(path)
            except Exception as e:
                print(f"  ✗ Error downloading {info['label']}: {e}")
        
        print(f"\n✓ Successfully downloaded {len(downloaded_paths)}/{len(image_infos)} images")
        print(f"  Saved to: {output_dir}")
        
        return downloaded_paths
    
    def download_single_image(self, image_url: str, output_path: str,
                             format: str = 'TIFF') -> Path:
        """
        Download a single IIIF image URL.
        
        Args:
            image_url: Direct URL to image
            output_path: Path to save image
            format: Image format
            
        Returns:
            Path to downloaded image
        """
        output_path = Path(output_path)
        print(f"Downloading image from: {image_url}")
        
        path = self.download_image(image_url, output_path, format, delay=0)
        
        print(f"✓ Image saved to: {path}")
        return path


def main():
    parser = argparse.ArgumentParser(
        description='Download images from IIIF manifests or URLs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download from config file
  python iiif_downloader.py yale_ms402_config.ini

  # Download from config with custom settings
  python iiif_downloader.py my_manuscript_config.ini

  # The config file should have [paths] section with either:
  #   iiif_url = <single image URL>
  #   OR
  #   iiif_manifest = <manifest URL>
  #   
  # Optional: cache_dir, max_images
        '''
    )
    
    parser.add_argument(
        'config',
        type=str,
        nargs='?',
        default='yale_ms402_config.ini',
        help='Path to configuration file (default: yale_ms402_config.ini)'
    )
    
    parser.add_argument(
        '--max-images',
        type=int,
        help='Override max_images from config (limit number of pages to download)'
    )
    
    parser.add_argument(
        '--format',
        choices=['TIFF', 'PNG', 'JPEG'],
        help='Override image format from config (default: TIFF)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = PipelineConfig(args.config)
        print("\n" + "="*60)
        print("IIIF Downloader")
        print("="*60)
        
        # Get IIIF settings from config
        paths = config.get_paths()
        
        # Check if IIIF URL is configured
        iiif_url = paths.get('iiif_url')
        if not iiif_url:
            print("\nError: No IIIF URL found in config.")
            print("Please add 'iiif_url' or 'iiif_manifest' to [paths] section.")
            print("\nExample:")
            print("  [paths]")
            print("  iiif_manifest = https://collections.library.yale.edu/manifests/10269778")
            print("  cache_dir = ../cache")
            return
        
        # Get settings
        cache_dir = paths.get('cache_dir', Path('../cache'))
        output_format = args.format if args.format else 'TIFF'
        
        # Get max_images from config or command line
        max_images = args.max_images
        if max_images is None and config.parser.has_option('iiif', 'max_images'):
            max_images = config.parser.getint('iiif', 'max_images')
        
        # Get delay from config
        delay = 0.5
        if config.parser.has_option('iiif', 'delay'):
            delay = config.parser.getfloat('iiif', 'delay')
        
        print(f"\nSettings:")
        print(f"  IIIF URL: {iiif_url}")
        print(f"  Cache dir: {cache_dir}")
        print(f"  Format: {output_format}")
        if max_images:
            print(f"  Max images: {max_images}")
        print(f"  Delay: {delay}s\n")
        
        # Create downloader
        downloader = IIIFDownloader(cache_dir=str(cache_dir))
        
        # Check if manifest or single image
        if downloader.is_manifest_url(iiif_url):
            # Download from manifest
            downloaded = downloader.download_all_images(
                manifest_url=iiif_url,
                output_dir=str(cache_dir),
                format=output_format,
                delay=delay,
                max_images=max_images
            )
            
            print(f"\n{'='*60}")
            print(f"✓ Download complete!")
            print(f"{'='*60}")
            print(f"\nDownloaded {len(downloaded)} images to: {cache_dir}")
            print(f"\nNext steps:")
            print(f"1. Run Kraken segmentation on the images")
            print(f"2. Update xml_file path in config")
            print(f"3. Run the pipeline scripts")
            
        else:
            # Download single image
            output_path = cache_dir / f"manuscript.{output_format.lower()}"
            downloader.download_single_image(
                image_url=iiif_url,
                output_path=str(output_path),
                format=output_format
            )
            
            print(f"\n{'='*60}")
            print(f"✓ Download complete!")
            print(f"{'='*60}")
            print(f"\nImage saved to: {output_path}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure the config file exists.")
    except ValueError as e:
        print(f"Configuration error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Download error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
