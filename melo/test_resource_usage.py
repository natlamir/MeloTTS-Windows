#this script requires you to use: pip install psutil GPUtil
import os
import time
import psutil
import GPUtil
import torch
from melo.api import TTS
import numpy as np

def get_system_usage():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    ram_usage = ram.used / (1024 ** 3)  # Convert to GB
    
    gpu_stats = None
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get first GPU
                gpu_stats = {
                    'gpu_load': gpu.load * 100,  # Convert to percentage
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal
                }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
    
    return {
        'cpu_percent': cpu_percent,
        'ram_used_gb': ram_usage,
        'gpu_stats': gpu_stats
    }

def run_tts_test(text, device, output_filename, speaker_id=0):
    """Run TTS test on specified device and monitor resources"""
    print(f"\nRunning TTS test on {device.upper()}...")
    
    # Clear CUDA cache and ensure clean state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize model on specified device
    model = TTS(language="EN", device=device)
    
    # Get initial resource usage
    initial_usage = get_system_usage()
    print(f"Initial resource usage:")
    print(f"CPU: {initial_usage['cpu_percent']:.1f}%")
    print(f"RAM: {initial_usage['ram_used_gb']:.1f} GB")
    if initial_usage['gpu_stats']:
        print(f"GPU: {initial_usage['gpu_stats']['gpu_load']:.1f}%")
        print(f"VRAM: {initial_usage['gpu_stats']['gpu_memory_used']:.1f} MB / "
              f"{initial_usage['gpu_stats']['gpu_memory_total']:.1f} MB")
    
    # Run TTS
    start_time = time.time()
    try:
        # Force BERT models to the correct device before TTS
        if device == "cuda":
            from melo.text.english_bert import model as bert_model
            if bert_model is not None:
                bert_model.to("cuda")
        
        audio = model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=output_filename,
            quiet=True
        )
        
        # Get peak resource usage
        peak_usage = get_system_usage()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nPeak resource usage:")
        print(f"CPU: {peak_usage['cpu_percent']:.1f}%")
        print(f"RAM: {peak_usage['ram_used_gb']:.1f} GB")
        if peak_usage['gpu_stats']:
            print(f"GPU: {peak_usage['gpu_stats']['gpu_load']:.1f}%")
            print(f"VRAM: {peak_usage['gpu_stats']['gpu_memory_used']:.1f} MB / "
                  f"{peak_usage['gpu_stats']['gpu_memory_total']:.1f} MB")
        
        print(f"\nConversion completed in {duration:.2f} seconds")
        print(f"Output saved to: {output_filename}")
        
    except Exception as e:
        print(f"Error during TTS conversion: {e}")
    
    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Create output directory
    os.makedirs("tts_outputs", exist_ok=True)
    
    # Test text
    text = "This is a test of the MeloTTS system in both GPU and CPU modes. How's the performance?"

    # Get starting resource usage (before using MeloTTS for inference)
    peak_usage = get_system_usage()
    
    print(f"\nPeak resource usage before MeloTTS inference:")
    print(f"--CPU: {peak_usage['cpu_percent']:.1f}%")
    print(f"--RAM: {peak_usage['ram_used_gb']:.1f} GB")
    if peak_usage['gpu_stats']:
        print(f"--GPU: {peak_usage['gpu_stats']['gpu_load']:.1f}%")
        print(f"--VRAM: {peak_usage['gpu_stats']['gpu_memory_used']:.1f} MB / "
                f"{peak_usage['gpu_stats']['gpu_memory_total']:.1f} MB")
    
    # Run CPU tests
    print("\n=== CPU Test - First Run (Warmup) ===")
    run_tts_test(
        text=text,
        device="cpu",
        output_filename="tts_outputs/cpu_output_1.wav"
    )
    
    print("\n=== CPU Test - Second Run (Measurement) ===")
    run_tts_test(
        text=text,
        device="cpu",
        output_filename="tts_outputs/cpu_output_2.wav"
    )

    print("\n=== CPU Test - Third Run (Extra Measurement) ===")
    run_tts_test(
        text=text,
        device="cpu",
        output_filename="tts_outputs/cpu_output_3.wav"
    )
    
    # Run GPU tests if available
    if torch.cuda.is_available():
        # Force CUDA synchronization and cache clear before GPU test
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        print("\n=== GPU Test - First Run (Warmup) ===")
        run_tts_test(
            text=text,
            device="cuda",
            output_filename="tts_outputs/gpu_output_1.wav"
        )
        
        print("\n=== GPU Test - Second Run (Measurement) ===")
        run_tts_test(
            text=text,
            device="cuda",
            output_filename="tts_outputs/gpu_output_2.wav"
        )

        print("\n=== GPU Test - Third Run (Extra Measurement) ===")
        run_tts_test(
            text=text,
            device="cuda",
            output_filename="tts_outputs/gpu_output_3.wav"
        )
    else:
        print("\nGPU not available for testing")

if __name__ == "__main__":
    main()
