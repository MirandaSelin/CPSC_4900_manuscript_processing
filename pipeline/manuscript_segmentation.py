#!/usr/bin/env python
"""
Manuscript Segmentation Script

This script runs Kraken OCR segmentation on manuscript images to detect text lines.
It wraps the Kraken command-line tool and uses configuration from an INI file.

Usage:
    python manuscript_segmentation.py gospel_config.ini
    
Requirements:
    - Kraken must be installed in your environment
    - Run this with the kraken-env virtual environment activated
    
The script will:
    1. Read paths from the config file
    2. Run: kraken -a -i input.tiff output.xml segment -bl
    3. Generate ALTO XML format segmentation with baseline detection
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
    
    print(f"\nRunning Kraken segmentation...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Input:  {tiff_path}")
    print(f"Output: {xml_output_path}")
    print("\nThis may take a minute...\n")
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Print Kraken's output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Kraken segmentation:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def main():
    """Run Kraken segmentation using config file."""
    
    parser = argparse.ArgumentParser(
        description='Run Kraken OCR segmentation on manuscript images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python manuscript_segmentation.py gospel_config.ini
  python manuscript_segmentation.py /path/to/my_manuscript_config.ini

Note:
  This script requires Kraken to be installed. Make sure you're running
  it in an environment where Kraken is available (e.g., kraken-env).
  
  To activate kraken-env:
    source ~/venvs/kraken-env/bin/activate
    
  Or if kraken-env is in your project:
    source kraken-env/bin/activate
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
            print("     source kraken-env/bin/activate")
            print("  2. Install Kraken:")
            print("     pip install kraken")
            sys.exit(1)
        
        # Get paths from config
        paths = config.get_paths()
        tiff_file = paths['tiff_file']
        xml_file = paths['xml_file']
        
        # Validate input file exists
        if not tiff_file.exists():
            print(f"\n❌ Error: Input TIFF file not found: {tiff_file}")
            sys.exit(1)
        
        # Check if output already exists
        if xml_file.exists():
            response = input(f"\n⚠️  Output file already exists: {xml_file}\n   Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Segmentation cancelled.")
                sys.exit(0)
        
        # Run segmentation
        success = run_segmentation(tiff_file, xml_file)
        
        if success:
            print(f"\n✓ Segmentation complete!")
            print(f"  Output saved to: {xml_file}")
            print(f"\nYou can now run the rest of the pipeline:")
            print(f"  python glyph_extraction.py {args.config}")
            print(f"  python glyph_embedder.py {args.config}")
            print(f"  python glyph_similarity_search.py {args.config}")
        else:
            print("\n❌ Segmentation failed. See errors above.")
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
