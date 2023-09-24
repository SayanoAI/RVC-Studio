> :warning: **The main branch is bleeding edge**: Expect frequent updates and many breaking changes after every commit

# RVC Studio
This project provides a comprehensive platform for training RVC models and generating AI voice covers. Use the app to download the required files before using or manually download them here: https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main

## Features
* Youtube music downloader: download any music video from Youtube as an mp3 file with just one click.
* 1-click AI song covers: easily create AI song covers using RVC.
* RVC Model fine-tuning: fine-tune an RVC model to mimic any voice you want using your own data.
* 1-click TTS using RVC model: convert any text to speech using the fine-tuned VC model with just one click.
* Built-in tensorboard: You can monitor the training progress and performance of your VC model using a built-in tensorboard dashboard.
* LLM integration: chat with your RVC model in real time using popular LLMs.
* Auto-Playlist: let your RVC model sing songs from your favourite playlist.

## Planned Features
* Demucs: Meta's vocals and instrumental music source separation.
* Audio-postprocessing: You can enhance the quality of your generated songs by adding reverbs, echos, etc.
* TTS using cloud API: use a cloud-based text-to-speech service to generate high-quality and natural-sounding speech from any text.
* Real-time VC interface: convert your voice using your favourite RVC model.

## Requirements
- Python 3.6 or higher (developed and tested on v3.8.17)
- Pip
- Virtualenv or conda package manager

## Easy Install
1. Clone this repository or download the zip file and extract it.
2. Double-click "conda-installer.bat" to install the latest version of [conda package manager](https://docs.conda.io/projects/miniconda/en/latest/)
3. Double-click "conda-start.bat" (if you skipped step 2.)

## Manual Installation
1. Clone this repository or download the zip file.
2. Navigate to the project directory and create a virtual environment with the command `virtualenv venv`.
3. Activate the virtual environment with the command `source venv/bin/activate` on Linux/Mac or `venv\Scripts\activate` on Windows. Or use `conda create -n RVC-Studio & conda activate RVC-Studio` if you're using conda package manager.
4. Install the required packages with the command `pip install -r requirements.txt`.
5. Run the streamlit app with the command `streamlit run Home.py`.

Or run it in [Google Colab](https://colab.research.google.com/github/SayanoAI/RVC-Studio/blob/master/RVC_Studio.ipynb)

## Instructions for Inference page
1. Download all the required models on the webui page or here: https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main
2. Put your favourite songs in the ./songs folder
3. Press "Refresh Data" button
4. Select a song (only wav/flac/ogg/mp3 are supported for now)
5. Select a voice model (put your RVC v2 models in ./models/RVC/ and index file in ./models/RVC/.index/)
6. Choose a vocal extraction model (preprocessing model is optional)
7. Click "Save Options" and "1-Click VC" to get started

## Instructions for Chat page
1. Download one of the following LLM (or use the homepage downloader):
* [airoboros-7B](https://huggingface.co/TheBloke/Airoboros-L2-7B-2.1-GGUF/blob/main/airoboros-l2-7b-2.1.Q4_K_M.gguf)
* [pygmalion-7B](https://huggingface.co/TheBloke/Pygmalion-2-7B-GGUF/blob/main/pygmalion-2-7b.Q4_K_M.gguf)
* [zarablend-7B](https://huggingface.co/TheBloke/Zarablend-MX-L2-7B-GGUF/blob/main/zarablend-mx-l2-7b.Q4_K_M.gguf)
* [Mythomax-L2-Kimiko-13B](https://huggingface.co/TheBloke/MythoMax-L2-Kimiko-v2-13B-GGUF/resolve/main/mythomax-l2-kimiko-v2-13b.Q4_K_M.gguf)
2. Write your name (this is what the LLM will call you)
3. Select Your Character or create one in the form below
4. Select a language model, you will have to set up the configuration yourself in the form below if you use your own models
5. Click "Start Chatting" to chat with your model

**Feel free to use larger versions of these models if your computer can handle it. (you will have to build your own config)**

## Dockerize
Run `docker compose up --build` in the main project folder.

~~**Known issue:** Tensorboard doesn't work inside a docker container. Feel free to submit a PR if you know a solution.~~ fixed in commit 8b720936b4dab347cba0e4a791330fb533bfdf1d 

## FAQs
* Trouble with ffmpeg/espeak? [Read this](/dist/README.md)

## Disclaimer
This project is for educational and research purposes only. The generated voice overs are not intended to infringe on any copyrights or trademarks of the original songs or text. The project does not endorse or promote any illegal or unethical use of the generative AI technology. The project is not responsible for any damages or liabilities arising from the use or misuse of the generated voice overs.

## Credits
This project uses code and AI models from the following repositories:

- [Retrieval-based Voice Conversion WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) by RVC-Project.
- [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) by Anjok07.
- [Streamlit](https://github.com/streamlit/streamlit) by streamlit.
- [Tacotron 2 - PyTorch implementation with faster-than-realtime inference](https://github.com/NVIDIA/tacotron2) by NVIDIA. 
- [Bark: A Speech Synthesis Toolkit for Bengali Language](https://github.com/suno-ai/bark) by suno-ai.
- [SpeechT5: A Self-Supervised Pre-training Model for Speech Recognition and Generation](https://github.com/microsoft/SpeechT5) by microsoft.
- [VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech](https://github.com/jaywalnut310/vits) by jaywalnut310 et al.
- [Applio-RVC-Fork](https://github.com/IAHispano/Applio-RVC-Fork) by IAHispano

We thank all the authors and contributors of these repositories for their amazing work and for making their code and models publicly available.