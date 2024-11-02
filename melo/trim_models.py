import os
import torch
import argparse
from pathlib import Path

def clean_checkpoint(input_path):
    """
    Convert a full MeloTTS training checkpoint to an inference-only checkpoint.
    
    Args:
        input_path (Path): Path to the full checkpoint
    """
    print(f"Processing checkpoint: {input_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(input_path, map_location='cpu')
    
    print("\nCheckpoint contents:")
    print("Keys in checkpoint:", list(checkpoint.keys()))
    
    # Create clean checkpoint keeping only necessary components
    clean_checkpoint = {
        'model': checkpoint['model'],  # Keep the model state
        'iteration': checkpoint['iteration'],  # Keep track of training iteration
        'learning_rate': checkpoint['learning_rate'],  # Keep the learning rate
        # Exclude the optimizer state
    }
    
    # Generate output path with _trimmed suffix
    output_path = input_path.parent / f"{input_path.stem}_trimmed{input_path.suffix}"
    
    # Save the clean checkpoint
    print(f"\nSaving clean checkpoint to: {output_path}")
    torch.save(clean_checkpoint, output_path)
    
    # Print size comparison
    original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
    clean_size = os.path.getsize(output_path) / (1024 * 1024)    # MB
    
    print(f"\nSize comparison:")
    print(f"Original checkpoint: {original_size:.2f} MB")
    print(f"Clean checkpoint: {clean_size:.2f} MB")
    print(f"Size reduction: {((original_size - clean_size) / original_size * 100):.1f}%")
    
    # Verify the clean checkpoint was saved properly
    if os.path.getsize(output_path) == 0:
        raise Exception("Error: Output file is empty!")
    
    # Load the clean checkpoint to verify it's valid
    try:
        test_load = torch.load(output_path, map_location='cpu')
        print("\nVerification successful - cleaned checkpoint contains:")
        print("Keys in cleaned checkpoint:", list(test_load.keys()))
    except Exception as e:
        raise Exception(f"Error verifying cleaned checkpoint: {str(e)}")

def should_process_file(filename):
    """
    Check if the file should be processed based on filename patterns.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file should be processed, False otherwise
    """
    excluded_prefixes = ['D_', 'DUR_']
    return not any(filename.startswith(prefix) for prefix in excluded_prefixes)

def process_directory(directory_path, debug=False):
    """
    Process all eligible .pth files in the given directory.
    
    Args:
        directory_path (str): Path to directory containing checkpoints
        debug (bool): Whether to print debug information
    """
    # Clean up the path string and convert to Path object
    directory_path = directory_path.strip().strip('"').strip("'")
    directory = Path(directory_path).resolve()
    
    if debug:
        print(f"Attempting to process directory: {directory}")
    
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    # Find all .pth files in the directory that don't start with excluded prefixes
    pth_files = [f for f in directory.glob("*.pth") if should_process_file(f.name)]
    
    if not pth_files:
        print(f"No eligible .pth files found in {directory}")
        return
    
    print(f"Found {len(pth_files)} eligible .pth files to process")
    
    # Print files that will be skipped if debug is enabled
    if debug:
        all_pth_files = list(directory.glob("*.pth"))
        skipped_files = [f for f in all_pth_files if not should_process_file(f.name)]
        if skipped_files:
            print("\nSkipping the following files:")
            for f in skipped_files:
                print(f"- {f.name}")
    
    success_count = 0
    for pth_file in pth_files:
        try:
            print(f"\nProcessing {pth_file.name}...")
            clean_checkpoint(pth_file)
            success_count += 1
        except Exception as e:
            print(f"Error processing {pth_file.name}: {str(e)}")
            if debug:
                import traceback
                traceback.print_exc()
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} out of {len(pth_files)} files")

def main():
    parser = argparse.ArgumentParser(description='Convert MeloTTS training checkpoints to inference-only checkpoints')
    parser.add_argument('directory', type=str, help='Directory containing checkpoint files')
    parser.add_argument('--debug', action='store_true', 
                       help='Print additional debugging information')
    
    args = parser.parse_args()
    
    try:
        process_directory(args.directory, args.debug)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()