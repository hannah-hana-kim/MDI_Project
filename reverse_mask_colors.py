import os
from PIL import Image, ImageOps
from pathlib import Path
import argparse

def reverse_mask_colors(input_dir, output_dir):
    """
    Reverse the colors in mask images (black <-> white)
    
    Args:
        input_dir: Path to directory containing mask images
        output_dir: Path to save reversed mask images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    mask_files = [f for f in input_path.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Found {len(mask_files)} mask images to process")
    
    for mask_file in mask_files:
        try:
            # Open the mask image
            mask = Image.open(mask_file)
            
            # Convert to grayscale if not already
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Invert the colors (black becomes white, white becomes black)
            inverted_mask = ImageOps.invert(mask)
            
            # Save the inverted mask
            output_file = output_path / mask_file.name
            inverted_mask.save(output_file)
            
            print(f"Processed: {mask_file.name} -> {output_file}")
            
        except Exception as e:
            print(f"Error processing {mask_file.name}: {e}")

def process_all_masked_data(base_data_dir, output_base_dir):
    """
    Process ALL masked data in your data structure, preserving exact folder structure
    Handles nested structure: category/object_type/numbered_folders/images
    
    Args:
        base_data_dir: Path to your 'data' folder
        output_base_dir: Path where reversed masks will be saved (data/masked_processed)
    """
    base_path = Path(base_data_dir)
    output_base = Path(output_base_dir)
    
    # Path to all masked frames
    masked_frames_path = base_path / "masked_frames"
    
    if not masked_frames_path.exists():
        print(f"Directory not found: {masked_frames_path}")
        return
    
    # Get all masked categories (furniture, vehicles, pets, etc.)
    masked_categories = [d for d in masked_frames_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(masked_categories)} masked categories:")
    for category in masked_categories:
        print(f"  - {category.name}")
    
    total_processed = 0
    
    # Process each masked category
    for category_dir in masked_categories:
        print(f"\n{'='*60}")
        print(f"Processing category: {category_dir.name}")
        print(f"{'='*60}")
        
        # Get object types (like 'chair', 'tricycle', etc.)
        object_types = [d for d in category_dir.iterdir() if d.is_dir()]
        
        if object_types:
            print(f"Found {len(object_types)} object types:")
            for obj_type in object_types:
                print(f"  - {obj_type.name}")
            
            for object_type_dir in object_types:
                print(f"\n  Processing object type: {object_type_dir.name}")
                
                # Check if this object type has numbered subfolders
                numbered_folders = [d for d in object_type_dir.iterdir() if d.is_dir()]
                
                if numbered_folders:
                    # Has numbered subfolders (01, 02, 03, etc.)
                    print(f"    Found {len(numbered_folders)} numbered folders: {[f.name for f in numbered_folders[:5]]}{'...' if len(numbered_folders) > 5 else ''}")
                    
                    for numbered_folder in numbered_folders:
                        # Count images in this numbered folder
                        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                        images_in_folder = [f for f in numbered_folder.iterdir() 
                                          if f.is_file() and f.suffix.lower() in image_extensions]
                        
                        if images_in_folder:
                            # Preserve exact structure: masked_processed/category/object_type/numbered_folder
                            output_numbered_dir = output_base / category_dir.name / object_type_dir.name / numbered_folder.name
                            
                            print(f"      Processing {numbered_folder.name}: {len(images_in_folder)} images")
                            reverse_mask_colors(numbered_folder, output_numbered_dir)
                            total_processed += len(images_in_folder)
                        else:
                            print(f"      {numbered_folder.name}: No images found")
                else:
                    # Images directly in object type folder (no numbered subfolders)
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                    images_count = len([f for f in object_type_dir.iterdir() 
                                      if f.is_file() and f.suffix.lower() in image_extensions])
                    
                    if images_count > 0:
                        output_object_dir = output_base / category_dir.name / object_type_dir.name
                        print(f"    Processing images directly: {images_count} images")
                        reverse_mask_colors(object_type_dir, output_object_dir)
                        total_processed += images_count
                    else:
                        print(f"    No images found in {object_type_dir.name}")
        else:
            # This category has images directly (no object type subfolders)
            print(f"Processing images directly in: {category_dir.name}")
            
            output_category_dir = output_base / category_dir.name
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            images_count = len([f for f in category_dir.iterdir() 
                              if f.is_file() and f.suffix.lower() in image_extensions])
            
            if images_count > 0:
                reverse_mask_colors(category_dir, output_category_dir)
                total_processed += images_count
            else:
                print(f"  No images found in {category_dir.name}")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE!")
    print(f"Total images processed: {total_processed}")
    print(f"Output directory: {output_base}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Reverse mask colors for inpainting")
    parser.add_argument("--input_dir", type=str, help="Input directory containing masks")
    parser.add_argument("--output_dir", type=str, help="Output directory for reversed masks")
    parser.add_argument("--data_dir", type=str, default="./data", 
                       help="Base data directory (default: ./data)")
    parser.add_argument("--process_all", action="store_true", 
                       help="Process ALL masked data automatically")
    
    args = parser.parse_args()
    
    if args.process_all:
        # Process ALL masked data, save to data/masked_processed
        output_dir = Path(args.data_dir) / "masked_processed"
        process_all_masked_data(args.data_dir, output_dir)
    elif args.input_dir and args.output_dir:
        # Process single directory
        reverse_mask_colors(args.input_dir, args.output_dir)
    else:
        print("Please provide either --input_dir and --output_dir, or use --process_all")
        print("\nExamples:")
        print("1. Process ALL masked data (recommended):")
        print("   python reverse_mask_colors.py --process_all")
        print("\n2. Process single category:")
        print("   python reverse_mask_colors.py --input_dir ./data/masked_frames/masked_furniture_frames/chair --output_dir ./data/masked_processed/masked_furniture_frames/chair")
        print("\n3. Custom data directory:")
        print("   python reverse_mask_colors.py --data_dir /path/to/your/data --process_all")

if __name__ == "__main__":
    main()