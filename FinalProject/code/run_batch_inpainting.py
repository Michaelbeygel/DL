#!/usr/bin/env python3
"""
Batch runner for inpainting scripts with multiple image numbers
"""
import subprocess
import sys
from pathlib import Path

def run_inpainting_batch(script_type="main", image_nums=range(1, 7)):
    """
    Run vanilla_inpainting.py or main_inpainting.py for multiple image numbers.
    
    Args:
        script_type: Either "vanilla" or "main" (default: "main")
        image_nums: Iterable of image numbers to process (default: 1-6)
    """
    if script_type == "vanilla":
        script_name = "vanilla_inpainting.py"
    elif script_type == "main":
        script_name = "main_inpainting.py"
    else:
        print(f"Error: Unknown script type '{script_type}'. Must be 'vanilla' or 'main'")
        sys.exit(1)
    
    script = Path(__file__).parent / script_name
    
    if not script.exists():
        print(f"Error: {script} not found")
        sys.exit(1)
    
    for image_num in image_nums:
        print(f"\n{'='*60}")
        print(f"Processing image {image_num}...")
        print(f"{'='*60}\n")
        
        try:
            # Run the script with image_num as argument
            result = subprocess.run(
                [sys.executable, str(script), "--image_num", str(image_num)],
                cwd=script.parent,
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
    script_type = sys.argv[1] if len(sys.argv) > 1 else "main"
    run_inpainting_batch(script_type=script_type)
