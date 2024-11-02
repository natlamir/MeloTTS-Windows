from melo.api import TTS
import sounddevice as sd
import numpy as np
from glob import glob
import os

def get_ckpt_files(folder_path):
    return [os.path.basename(f) for f in glob(os.path.join(folder_path, '*.pth'))]

def get_config_path(ckpt_path):
    """Get the corresponding config path for a checkpoint file"""
    base_path = os.path.splitext(ckpt_path)[0]
    config_options = [
        f"{base_path}.config.json",  # try g_1000.config.json
        f"{base_path}.json",         # try g_1000.json
    ]
    
    for config_path in config_options:
        if os.path.exists(config_path):
            return config_path
    return None

def load_custom_model(ckpt_file):
    ckpt_path = os.path.join("custom", ckpt_file)
    config_path = get_config_path(ckpt_path)
    
    if config_path is None:
        print(f"Error: No config file found for {ckpt_file}")
        print("Expected either:")
        print(f"- {os.path.splitext(ckpt_file)[0]}.config.json")
        print(f"- {os.path.splitext(ckpt_file)[0]}.json")
        return None
        
    try:
        return TTS(language="EN", config_path=config_path, ckpt_path=ckpt_path)
    except Exception as e:
        print(f"Error loading custom model: {e}")
        return None


def get_speaker_names(model):
    """Get mapping of speaker IDs to names"""
    return model.hps.data.spk2id

def list_available_models():
    models = {
        'EN': TTS(language='EN'),
        'ES': TTS(language='ES'),
        'FR': TTS(language='FR'),
        'ZH': TTS(language='ZH'),
        'JP': TTS(language='JP'),
        'KR': TTS(language='KR'),
    }
    return models

def select_speaker(model):
    """Let user select a speaker from available options"""
    speakers = get_speaker_names(model)
    
    print("\nAvailable speakers:")
    for name, id in speakers.items():
        print(f"- {name} (ID: {id})")
    
    while True:
        speaker = input("\nEnter speaker name or ID: ")
        # Check if input is a number
        if speaker.isdigit():
            id_num = int(speaker)
            # Check if ID exists in values
            for name, spk_id in speakers.items():
                if spk_id == id_num:
                    return id_num
            print("Invalid speaker ID. Please try again.")
        # Check if input is a speaker name
        elif speaker in speakers:
            return speakers[speaker]
        print("Invalid input. Enter either speaker name or ID number.")

def main():
    print("Loading models...")
    models = list_available_models()
    
    # Check for custom models
    custom_folder = os.path.join("custom")
    custom_models = get_ckpt_files(custom_folder)
    
    print("\nAvailable languages:")
    for lang in models.keys():
        print(f"- {lang}")
    
    if custom_models:
        print("\nAvailable custom models:")
        for model in custom_models:
            print(f"- {model}")
    
    use_custom = input("\nUse custom model? (y/n): ").lower() == 'y'
    
    if use_custom:
        if not custom_models:
            print("No custom models found in 'custom' directory")
            return
            
        print("\nSelect custom model:")
        for i, model in enumerate(custom_models):
            print(f"{i+1}. {model}")
        
        try:
            choice = int(input("Enter number: ")) - 1
            model = load_custom_model(custom_models[choice])
            if not model:
                return
            speaker_id = select_speaker(model)
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    else:
        lang = input("\nSelect language (EN/ES/FR/ZH/JP/KR): ").upper()
        if lang not in models:
            print("Invalid language selected")
            return
        
        model = models[lang]
        speaker_id = select_speaker(model)
    
    print("\nEnter text to speak (Type 'quit' to exit)")
    while True:
        text = input("\nText (Type 'quit' to exit): ")
        if text.lower() == 'quit':
            break
            
        try:
            audio = model.tts_to_file(text, speaker_id, pbar=None, quiet=True)
            sd.play(audio, model.hps.data.sampling_rate)
            sd.wait()
            
        except Exception as e:
            print(f"Error generating speech: {e}")

if __name__ == "__main__":
    main()
