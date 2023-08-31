import numpy as np
import torch
import os

from num2words import num2words

from webui_utils import MAX_INT16, remix_audio

CWD = os.getcwd()
speecht5_checkpoint = "microsoft/speecht5_tts"
speecht5_vocoder_checkpoint = "microsoft/speecht5_hifigan"
bark_checkpoint = "suno/bark-small"
bark_voice_presets="v2/en_speaker_2"
tacotron2_checkpoint = "speechbrain/tts-tacotron2-ljspeech"
hifigan_checkpoint = "speechbrain/tts-hifigan-ljspeech"
embedding_checkpoint = "speechbrain/spkrec-xvect-voxceleb"
os.makedirs(os.path.join(CWD,"models","tts","embeddings"),exist_ok=True)
TTS_MODELS_DIR = os.path.join(CWD,"models","tts")

def __speecht5__(text, speaker_embedding=None, device="cpu"):
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    tts_vocoder = SpeechT5HifiGan.from_pretrained(speecht5_vocoder_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_vocoder_checkpoint),device_map=device)
    tts_processor = SpeechT5Processor.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    inputs = tts_processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :tts_model.config.max_text_positions]
    speech = tts_model.generate_speech(input_ids.to(device), speaker_embedding.to(device), vocoder=tts_vocoder)
    speech = (speech.cpu().numpy() * MAX_INT16).astype(np.int16)
    return speech, 16000

def cast_to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(e)
        return tensor
    
def __bark__(text, device="cpu"):
    from transformers import AutoProcessor, BarkModel
    bark_processor = AutoProcessor.from_pretrained(
        bark_checkpoint,
        cache_dir=os.path.join(TTS_MODELS_DIR,bark_checkpoint),
        torch_dtype=torch.float16)
    bark_model = BarkModel.from_pretrained(
        bark_checkpoint,
        cache_dir=os.path.join(TTS_MODELS_DIR,bark_checkpoint),
        torch_dtype=torch.float16).to(device)
    # bark_model.enable_cpu_offload()

    inputs = bark_processor(
    text=[text],
    return_tensors="pt",
    voice_preset=bark_voice_presets
    )
    tensor_dict = {k: cast_to_device(v,device) if hasattr(v,"to") else v for k, v in inputs.items()}
    speech_values = bark_model.generate(**tensor_dict, do_sample=True)
    sampling_rate = bark_model.generation_config.sample_rate
    speech = (speech_values.cpu().numpy().squeeze() * MAX_INT16).astype(np.int16)
    return speech, sampling_rate

def __tacotron2__(text, device="cpu"):
    from speechbrain.pretrained import Tacotron2
    from speechbrain.pretrained import HIFIGAN
    hifi_gan = HIFIGAN.from_hparams(source=hifigan_checkpoint, savedir=os.path.join(TTS_MODELS_DIR,hifigan_checkpoint), run_opts={"device": device})
    tacotron2 = Tacotron2.from_hparams(source=tacotron2_checkpoint, savedir=os.path.join(TTS_MODELS_DIR,tacotron2_checkpoint), run_opts={"device": device})
    # Running the TTS
    mel_output, _, _ = tacotron2.encode_text(text)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)
    speech = (waveforms.cpu().numpy().squeeze() * MAX_INT16).astype(np.int16)

    # return as numpy array
    return speech, 22050

def train_speaker_embedding(speaker: str,input_audio=None):
    embedding_path = f"./models/tts/embeddings/{speaker}.npy"
    if os.path.exists(embedding_path):
        speaker_embedding = np.load(embedding_path)
        speaker_embedding = torch.tensor(speaker_embedding)
        return speaker_embedding
    if input_audio is None: raise ValueError(f"Please train a speaker model, {embedding_path} is missing!")
    from speechbrain.pretrained import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(source=embedding_checkpoint, savedir=os.path.join(TTS_MODELS_DIR,embedding_checkpoint))
    audio,_ = remix_audio(input_audio,target_sr=16000,norm=True,resample=True,to_mono=True)
    embeddings = classifier.encode_batch(torch.from_numpy(audio),normalize=True).squeeze(0)
    np.save(f"./models/tts/embeddings/{speaker}.npy",embeddings.numpy())
    return embeddings

def parseText(text):
    words = text.split()
    words = [num2words(word) if word.isdigit() else word for word in words]
    return ' '.join(words)

def generate_speech(text, speaker=None, method="speecht5",device="cpu"):
    if text and len(text.strip()) == 0:
        return (np.zeros(0).astype(np.int16),16000)
    text = parseText(text) #convert numbers to words

    if method=="speecht5":
        if speaker is None: raise ValueError(f"Must provider a speaker_embedding for {method} inference!")
        if type(speaker)==str:
            speaker_embedding = np.load(speaker)
            speaker_embedding = torch.tensor(speaker_embedding)
        else: speaker_embedding = speaker
        return __speecht5__(text,speaker_embedding,device)
    elif method=="bark":
        return __bark__(text,device)
    elif method=="tacotron2":
        return __tacotron2__(text,device)
    else: return None