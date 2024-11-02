import torch
import gc
from melo.api import TTS
import os
import melo.text.english_bert as bert
import psutil
import GPUtil

def get_memory_usage():
    """Get current RAM and VRAM usage"""
    ram = psutil.virtual_memory()
    ram_usage = ram.used / (1024 ** 3)  # Convert to GB
    
    gpu_memory = None
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get first GPU
                gpu_memory = {
                    'used': gpu.memoryUsed,  # MB
                    'total': gpu.memoryTotal  # MB
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
    
    return ram_usage, gpu_memory

def clear_gpu_memory():
    """Clear both GPU and RAM memory more aggressively"""
    # First clear CUDA memory
    if torch.cuda.is_available():
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    if obj.is_cuda:
                        del obj
            except Exception:
                pass
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    # Force Python garbage collection
    gc.collect()
    
    # Optional: Force more aggressive garbage collection
    for _ in range(2):
        gc.collect()
    
    # Print memory stats to verify clearing
    ram = psutil.virtual_memory()
    print(f"\nAfter clearing - RAM Usage: {ram.used / (1024 ** 3):.2f} GB")
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"After clearing - VRAM Usage: {gpu.memoryUsed:.2f} MB / {gpu.memoryTotal:.2f} MB")

def reset_bert_model(device):
    """Reset the global BERT model in english_bert.py only when switching devices"""
    current_device = None
    
    # Print memory usage before reset
    ram_before, vram_before = get_memory_usage()
    print(f"\nMemory before BERT reset:")
    print(f"RAM Usage: {ram_before:.2f} GB")
    if vram_before:
        print(f"VRAM Usage: {vram_before['used']:.2f} MB / {vram_before['total']:.2f} MB")
    
    # Check current device of BERT model if it exists
    if bert.model is not None:
        current_device = next(bert.model.parameters()).device.type
        
        # If the model is already on the correct device, return early
        if (current_device == 'cuda' and device == 'cuda') or \
           (current_device == 'cpu' and device == 'cpu'):
            return
        
        # Explicitly move model to CPU before deletion if it's on CUDA
        if current_device == 'cuda':
            bert.model.cpu()
        
        # Delete the model and clear memory
        del bert.model
        clear_gpu_memory()
        bert.model = None
    
    # Create proper dummy input for BERT initialization
    dummy_text = "Hello world"
    tokens = bert.tokenizer(dummy_text, return_tensors="pt")
    word2ph = [1] * tokens["input_ids"].shape[1]
    _ = bert.get_bert_feature(dummy_text, word2ph, device)
    
    # Print memory usage after reset
    ram_after, vram_after = get_memory_usage()
    print(f"\nMemory after BERT reset:")
    print(f"RAM Usage: {ram_after:.2f} GB")
    if vram_after:
        print(f"VRAM Usage: {vram_after['used']:.2f} MB / {vram_after['total']:.2f} MB")
    
    print(f"BERT model reloaded: {current_device} -> {device}")

import time  # Add this to your imports

def generate_speech(model, text, spk_id, device, output_path):
    """Generate speech with specified device"""
    try:
        # Print memory usage before model movement
        ram_before, vram_before = get_memory_usage()
        print(f"\nMemory before moving model to {device}:")
        print(f"RAM Usage: {ram_before:.2f} GB")
        if vram_before:
            print(f"VRAM Usage: {vram_before['used']:.2f} MB / {vram_before['total']:.2f} MB")
        
        # If moving from CUDA to CPU, first move model to CPU then clear CUDA memory
        if device == "cpu" and next(model.parameters()).is_cuda:
            model.cpu()
            clear_gpu_memory()
        else:
            model = model.to(device)
            
        reset_bert_model(device)
        spk_id = torch.tensor([spk_id], device=device)
        
        # Print memory usage after model movement
        ram_after, vram_after = get_memory_usage()
        print(f"\nMemory after moving model to {device}:")
        print(f"RAM Usage: {ram_after:.2f} GB")
        if vram_after:
            print(f"VRAM Usage: {vram_after['used']:.2f} MB / {vram_after['total']:.2f} MB")
        
        # Add timing measurement
        start_time = time.time()
        model.tts_to_file(text, spk_id, output_path)
        end_time = time.time()
        conversion_time = end_time - start_time
        print(f"\n----Text-to-speech conversion time on {device}: {conversion_time:.2f} seconds")
        
    finally:
        clear_gpu_memory()


def alternate_gpu_cpu_inference(ckpt_path, text, language="EN", output_dir="outputs"):
    """Alternate between GPU and CPU inference"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    
    # Sequence of devices to test
    devices = ["cuda", "cpu", "cuda", "cpu"]
    current_model = None
    current_device = None
    
    for i, device in enumerate(devices):
        # Skip if CUDA not available for GPU inference
        if device == "cuda" and not torch.cuda.is_available():
            print(f"CUDA not available, skipping GPU inference {i+1}")
            continue
            
        print(f"\nGenerating speech using {device.upper()} - Round {i+1}")
        
        # Only create new model instance if switching devices or first run
        if current_model is None or current_device != device:
            if current_model is not None:
                del current_model
                clear_gpu_memory()
            
            current_model = TTS(language=language, config_path=config_path, 
                              ckpt_path=ckpt_path, device=device)
            current_device = device
        
        # Get first speaker ID
        spk_id = list(current_model.hps.data.spk2id.values())[0]
        
        output_path = os.path.join(output_dir, f"output_{i+1}_{device}.wav")
        
        # Generate speech
        generate_speech(current_model, text, spk_id, device, output_path)
        print(f"Generated: {output_path}")

if __name__ == "__main__":
    # Example usage
    ckpt_path = "C:/Users/lyria/Documents/Scripts/python/melotts/melotts/MeloTTS-Windows/melo/data/example_fused3voices_7wavs/output_3voices_7wavs/G_2000.pth"
    text = "This is a test of alternating between GPU and CPU inference."
    
    alternate_gpu_cpu_inference(ckpt_path, text)
