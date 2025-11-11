#!/usr/bin/env python
"""
Manuscript Segmentation Script

This script runs Kraken OCR segmentation on manuscript images to detect text lines.
It wraps the Kraken command-line tool and uses configuration from an INI file.

Usage:
    python manuscript_segmentation.py my_config.ini
    
Requirements:
    - Kraken must be installed in your environment
    - Run this with the kraken-env virtual environment activated
    - cache_dir must be specified in config with TIFF images
    
The script will:
    1. Read cache_dir from the config file
    2. Find all TIFF images in cache_dir
    3. Run Kraken on each: kraken -a -i input.tiff output.xml segment -bl
    4. Save XML files to cache_dir/segmentation/
    5. Generate ALTO XML format segmentation with baseline detection
"""

import argparse
import subprocess
import sys
from pathlib import Path
from config import PipelineConfig


def check_kraken_installed():
    """Check if Kraken is installed and accessible."""
    try:
        result = subprocess.run(
            ['kraken', '--version'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✓ Kraken found: {version}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def run_segmentation(tiff_path: Path, xml_output_path: Path) -> bool:
    """
    Run Kraken segmentation on the manuscript image.
    
    Args:
        tiff_path: Path to input TIFF image
        xml_output_path: Path where ALTO XML should be saved
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure output directory exists
    xml_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build the Kraken command
    # kraken -a -i input.tiff output.xml segment -bl
    cmd = [
        'kraken',
        '-a',  # Output ALTO XML format
        '-i', str(tiff_path), str(xml_output_path),  # Input/output files
        'segment',  # Segmentation mode
        '-bl'  # Detect baselines (text lines)
    ]
    
    print(f"Running Kraken segmentation...")
    print(f"  Input:  {tiff_path.name}")
    print(f"  Output: {xml_output_path.name}")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print Kraken's output (if any)
        if result.stdout and result.stdout.strip():
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error running Kraken segmentation:")
        print(f"  Return code: {e.returncode}")
        if e.stderr:
            print(f"  STDERR: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def find_all_downloaded_images(cache_dir: Path) -> list[Path]:
    """
    Find all TIFF images in the cache directory.
    
    Args:
        cache_dir: Directory containing downloaded IIIF images
        
    Returns:
        Sorted list of TIFF image paths
    """
    tiff_files = sorted(cache_dir.glob("*.tiff")) + sorted(cache_dir.glob("*.tif"))
    return sorted(set(tiff_files))  # Remove duplicates and sort


def main():
    """Run Kraken segmentation using config file."""
    
    parser = argparse.ArgumentParser(
        description='Run Kraken OCR segmentation on manuscript images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python manuscript_segmentation.py gospel_config.ini
  python manuscript_segmentation.py yale_ms402_config.ini

Note:
  This script requires Kraken to be installed and processes ALL TIFF files
  in the cache_dir specified in your config file.
  
  For IIIF sources: Run iiif_downloader.py first to populate cache_dir
  For local files: Copy your TIFF files to cache_dir
  
  The script will create XML files in cache_dir/segmentation/ for each image.
  
  To activate kraken-env:
    source /Users/miranda/Documents/Yale/Classes/CPSC_490/kraken-env/bin/activate
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
        
        # Check if Kraken is installed
        if not check_kraken_installed():
            print("\n❌ Error: Kraken is not installed or not in PATH")
            print("\nTo install Kraken:")
            print("  1. Activate your kraken environment:")
            print("     source /Users/miranda/Documents/Yale/Classes/CPSC_490/kraken-env/bin/activate")
            print("  2. Install Kraken:")
            print("     pip install kraken")
            sys.exit(1)
        
        # Get paths from config
        paths = config.get_paths()
        
        # Get cache directory
        cache_dir = paths.get('cache_dir')
        if not cache_dir:
            print("\n❌ Error: cache_dir not specified in config file")
            print("\nPlease specify cache_dir in the [paths] section of your config.")
            sys.exit(1)
        
        # Find all TIFF images in cache directory
        image_files = find_all_downloaded_images(cache_dir)
        
        if not image_files:
            print(f"\n❌ Error: No TIFF images found in cache directory: {cache_dir}")
            print("\nFor IIIF sources: Run iiif_downloader.py first to download images.")
            print("For local files: Copy your TIFF files to the cache directory.")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Found {len(image_files)} image(s) to segment")
        print(f"{'='*60}\n")
        
        # Create segmentation output directory
        seg_dir = cache_dir / 'segmentation'
        seg_dir.mkdir(exist_ok=True)
        
        # Process each image
        successful = 0
        failed = 0
        
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] {image_path.name}")
            
            # Create output XML filename based on image filename
            xml_filename = image_path.stem + '.xml'
            xml_output = seg_dir / xml_filename
            
            # Run segmentation
            if run_segmentation(image_path, xml_output):
                print(f"  ✓ Saved: {xml_output.name}\n")
                successful += 1
            else:
                print(f"  ✗ Failed\n")
                failed += 1
        
        # Print summary
        print(f"{'='*60}")
        print(f"Segmentation Summary")
        print(f"{'='*60}")
        print(f"  Successful: {successful}/{len(image_files)}")
        print(f"  Failed:     {failed}/{len(image_files)}")
        print(f"  Output dir: {seg_dir}")
        print(f"{'='*60}\n")
        
        if successful > 0:
            print(f"✓ Segmentation complete!")
            print(f"\nNext steps:")
            if len(image_files) == 1:
                print(f"1. Update xml_file in config to: {seg_dir / image_files[0].stem}.xml")
                print(f"2. Run: python glyph_extraction.py {args.config}")
            else:
                print(f"1. Update xml_file in config to point to one of the XML files")
                print(f"2. Or run glyph_extraction.py on each XML file separately")
                print(f"\nExample:")
                print(f"  python glyph_extraction.py {args.config} --xml-file {seg_dir / image_files[0].stem}.xml")
        
        if failed > 0:
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSegmentation interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
