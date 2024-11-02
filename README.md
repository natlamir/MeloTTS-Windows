# Install Steps on Windows

1. Clone the repository
```
git clone https://github.com/natlamir/MeloTTS-Windows.git
cd MeloTTS-Windows
```

2. Create conda environment and install dependencies
```
conda env create -f environment.yml
conda activate melotts-win
pip install -e .
python -m unidic download
```
If you have trouble doing the download with the `python -m unidic download` you can try this:

- Download the zip from: https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip
- Place it in: C:\Users\YOUR_USER_ID\miniconda3\envs\melotts-win\Lib\site-packages\unidic
- Rename it to unidic.zip
- Replace the downalod.py file in this same directory with the one from https://github.com/natlamir/ProjectFiles/blob/main/melotts/download.py
- Now re-run the `python -m unidic download`. This info originally gotten from: https://github.com/myshell-ai/MeloTTS/issues/62#issuecomment-2067361999

3. Install pytorch
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Prepare faster-whisper (optional for fast transcribing of audio files):
   - Download cuda/cublas here [https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA11_win_v2.7z](https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA11_win_v2.7z), extract and place the 5 dll files directly into the `MeloTTS-Windows/melo/` folder
   - To install faster-whisper (and prevent conflicts with it) run this from the conda window:
```
pip install faster-whisper==0.9.0
pip install transformers==4.30.2 huggingface_hub==0.16.4
```

4. Run using:
```
melo-ui
```

# Local Training on Windows
## Preparing Dataset
1. In the `melo/data/example` folder, delete the example `metadata.list` file.
2. MeloTTS expects wav audio files (with a sample rate of 44100Hz). If you need to convert audio to wav format (with 44100Hz sample rate), create a folder called `audio` in the example folder and copy all your audio files into the `audio` folder
4. With a conda window activated with the enviroment open in the `melo` folder, run `ConvertAudiotoWav.bat` from the conda prompt. This will create a folder `data/example/wavs` with all of the converted wav files.
5. Create a transcript file by running `transcript_fast.bat` which will create a `data/example/metadata.list` file using faster-whisper. Alternately, you can run `python transcript.py` to use the original whisper.
6. Run `python preprocess_text.py --metadata data/example/metadata.list` to create the `train.list`, `config.json`, among other files in the `data/example` folder.
7. Modify `config.json` to change the batch size, epochs, learning rate, etc.
  - ⚠️ **Important, If you plan to Resume Training Later:**
    - The `eval_interval` setting determines how frequently your model is saved during training
    - For example, if `eval_interval=1000`, the model saves only once every 1000 steps
    - If you stop training between save points, any progress since the last save will be lost
    - For safer training sessions that you may need to resume later, use a smaller `eval_interval` value
    - You can also adjust `n_ckpts_to_keep` to limit the max models kept (if `n_ckpts_to_keep=5`, it will delete the oldest models when their are more than 5 saved models)
## Start Training
1. From the conda prompt run `train.bat` to start the training.
2. File will be created within the `data/example/config` folder with the checkpoints and other logging information.
3. To test out a checkpoint, run: `python infer.py --text "this is a test" -m "C:\ai\MeloTTS-Windows\melo\data\example\config\G_0.pth" -o output` changing the G_0 to the checkpoint you want to test with G_1000, G2000, etc.
4. When you want to use a checkpoint from the UI, create a `melo/custom` folder and copy the .pth and `config.json` file over from the `data/example/config`, rename the .pth to a user-friendly name, and launch the UI to see it in the custom voice dropdown.
5. To see the tensorboard, install `pip install tensorflow`
6. Run `tensorboard --logdir=data\example\config`
7. This will give you the local URL to view the tensorboard.
## Resuming Training
1. From the conda prompt run `train.bat` again to resume the training. The training will resume from the newest G_XXXX.pth file.
## Trimming Model
You can trim your model to make it a way smaller filesize (which will make it load faster during the model loading process). When testing, this made the model filesize about 66% smaller. Note the created trimmed model is for inference-only(using the model just to generate audio from text) and you won't be able to train it further.
1. Open `trim_models.bat` file in a text editor to change the directory to your G_XXXX.pth files and the save location, save the changes, then run `trim_models.bat` to create a trimmed model for inference only.

# Original Readme:
<div align="center">
  <div>&nbsp;</div>
  <img src="logo.png" width="300"/> 
</div>

## Introduction
MeloTTS is a **high-quality multi-lingual** text-to-speech library by [MIT](https://www.mit.edu/) and [MyShell.ai](https://myshell.ai). Supported languages include:

| Language | Example |
| --- | --- |
| English (American)    | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-US/speed_1.0/sent_000.wav) |
| English (British)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-BR/speed_1.0/sent_000.wav) |
| English (Indian)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN_INDIA/speed_1.0/sent_000.wav) |
| English (Australian)  | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-AU/speed_1.0/sent_000.wav) |
| English (Default)     | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/en/EN-Default/speed_1.0/sent_000.wav) |
| Spanish               | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/es/ES/speed_1.0/sent_000.wav) |
| French                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/fr/FR/speed_1.0/sent_000.wav) |
| Chinese (mix EN)      | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/zh/ZH/speed_1.0/sent_008.wav) |
| Japanese              | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/jp/JP/speed_1.0/sent_000.wav) |
| Korean                | [Link](https://myshell-public-repo-host.s3.amazonaws.com/myshellttsbase/examples/kr/KR/speed_1.0/sent_000.wav) |

Some other features include:
- The Chinese speaker supports `mixed Chinese and English`.
- Fast enough for `CPU real-time inference`.

## Usage
- [Use without Installation](docs/quick_use.md)
- [Install and Use Locally](docs/install.md)
- [Training on Custom Dataset](docs/training.md)

The Python API and model cards can be found in [this repo](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md#python-api) or on [HuggingFace](https://huggingface.co/myshell-ai).

## Join the Community

**Discord**

Join our [Discord community](https://discord.gg/myshell) and select the `Developer` role upon joining to gain exclusive access to our developer-only channel! Don't miss out on valuable discussions and collaboration opportunities.

**Contributing**

If you find this work useful, please consider contributing to this repo.

- Many thanks to [@fakerybakery](https://github.com/fakerybakery) for adding the Web UI and CLI part.

## Authors

- [Wenliang Zhao](https://wl-zhao.github.io) at Tsinghua University
- [Xumin Yu](https://yuxumin.github.io) at Tsinghua University
- [Zengyi Qin](https://www.qinzy.tech) at MIT and MyShell

**Citation**
```
@software{zhao2024melo,
  author={Zhao, Wenliang and Yu, Xumin and Qin, Zengyi},
  title = {MeloTTS: High-quality Multi-lingual Multi-accent Text-to-Speech},
  url = {https://github.com/myshell-ai/MeloTTS},
  year = {2023}
}
```

## License

This library is under MIT License, which means it is free for both commercial and non-commercial use.

## Acknowledgements

This implementation is based on [TTS](https://github.com/coqui-ai/TTS), [VITS](https://github.com/jaywalnut310/vits), [VITS2](https://github.com/daniilrobnikov/vits2) and [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2). We appreciate their awesome work.
