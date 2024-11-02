import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from faster_whisper import WhisperModel

def transcribe_folder(input_folder, output_file):
    LANGUAGE_CODE = "EN"
    LANGUAGE_MODEL = "EN-default"

    # Initialize the faster-whisper model
    model = WhisperModel("medium", device="cuda", compute_type="int8_float16")

    # Get the list of WAV files in the input directory
    wav_files = [file for file in os.listdir(input_folder) if file.endswith(".wav")]

    # Sort the WAV files in numeric order
    wav_files = sorted(wav_files, key=lambda x: int(os.path.splitext(x)[0]))

    # Open a text file for writing the transcripts
    with open(output_file, "w", encoding="utf-8") as transcript_file:
        # Prepare the output path with forward slashes
        output_path = input_folder.replace('\\', '/')
        
        # Iterate through each WAV file
        for i, wav_file in enumerate(wav_files):
            print(f"Transcribing: {wav_file}")
            wav_path = os.path.join(input_folder, wav_file)
            
            segments, info = model.transcribe(wav_path)
            transcribed_text = " ".join([segment.text for segment in segments]).strip()
            
            line = f"{output_path}/{wav_file}|{LANGUAGE_MODEL}|{LANGUAGE_CODE}|{transcribed_text}"
            if i < len(wav_files) - 1:
                line += "\n"
            transcript_file.write(line)

    print(f"Transcription complete. Check '{output_file}' for results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transcribe WAV files using faster-whisper')
    parser.add_argument('--wavs_path', type=str, default="data\\example\\wavs",
                      help='Path to the directory containing WAV files (default: data\\example\\wavs)')
    parser.add_argument('--metadata_path', type=str, default="data\\example\\metadata.list",
                      help='Path for the output metadata.list file (default: data\\example\\metadata.list)')
    
    args = parser.parse_args()
    
    # Check if the input directory exists
    if not os.path.exists(args.wavs_path):
        print(f"Error: Directory '{args.wavs_path}' does not exist.")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.metadata_path), exist_ok=True)
        
    # Run transcription
    transcribe_folder(args.wavs_path, args.metadata_path)
