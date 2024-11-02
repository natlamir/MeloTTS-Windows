@echo off
setlocal enabledelayedexpansion

rem Specify input and output folders here
set "INPUT_FOLDER=data\example\audio"
set "OUTPUT_FOLDER=data\example\wavs"

rem Create output folder if it doesn't exist
if not exist "%OUTPUT_FOLDER%" mkdir "%OUTPUT_FOLDER%"

rem Initialize counter for sequential naming
set /a counter=1

rem Loop through common audio formats that ffmpeg supports
for %%F in (
    "%INPUT_FOLDER%\*.mp3" 
    "%INPUT_FOLDER%\*.m4a" 
    "%INPUT_FOLDER%\*.wav" 
    "%INPUT_FOLDER%\*.ogg" 
    "%INPUT_FOLDER%\*.flac" 
    "%INPUT_FOLDER%\*.aac" 
    "%INPUT_FOLDER%\*.wma"
    "%INPUT_FOLDER%\*.aiff"
    "%INPUT_FOLDER%\*.aifc"
    "%INPUT_FOLDER%\*.opus"
    "%INPUT_FOLDER%\*.ape"
    "%INPUT_FOLDER%\*.wv"
    "%INPUT_FOLDER%\*.m4b"
    "%INPUT_FOLDER%\*.mp2"
    "%INPUT_FOLDER%\*.mp4"
    "%INPUT_FOLDER%\*.mpc"
    "%INPUT_FOLDER%\*.mka"
    "%INPUT_FOLDER%\*.ac3"
    "%INPUT_FOLDER%\*.dts"
    "%INPUT_FOLDER%\*.amr"
    "%INPUT_FOLDER%\*.au"
    "%INPUT_FOLDER%\*.mid"
) do (
    rem Convert the file using ffmpeg
    ffmpeg -i "%%F" -acodec pcm_s16le -ar 44100 "%OUTPUT_FOLDER%\!counter!.wav"
    
    rem Increment the counter
    set /a counter+=1
)

echo Conversion complete. Check the '%OUTPUT_FOLDER%' folder for the converted files.
