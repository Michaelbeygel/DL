#!/usr/bin/env python3
"""
Batch runner for main_inpainting.py with multiple image numbers
"""
import subprocess
import sys
from pathlib import Path

def run_inpainting_batch(image_nums=range(1, 8)):
    """
    Run main_inpainting.py for multiple image numbers.
    
    Args:
        image_nums: Iterable of image numbers to process (default: 1-7)
    """
    main_script = Path(__file__).parent / "main_inpainting.py"
    
    if not main_script.exists():
        print(f"Error: {main_script} not found")
        sys.exit(1)
    
    for image_num in image_nums:
        print(f"\n{'='*60}")
        print(f"Processing image {image_num}...")
        print(f"{'='*60}\n")
        
        try:
            # Run the script with image_num as argument
            result = subprocess.run(
                [sys.executable, str(main_script), "--image_num", str(image_num)],
                cwd=main_script.parent,
                check=True
            )
            print(f"✓ Image {image_num} completed successfully\n")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing image {image_num}: {e}\n")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All images processed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    run_inpainting_batch()
