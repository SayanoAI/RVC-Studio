import hashlib
import json
import numpy as np
import torch
import os
from lib.infer_pack.text.cleaners import english_cleaners
from lib.slicer2 import Slicer
from webui import get_cwd

from webui.audio import MAX_INT16, load_input_audio, remix_audio
from webui.downloader import BASE_CACHE_DIR, download_file

CWD = get_cwd()
    
speecht5_checkpoint = "microsoft/speecht5_tts"
speecht5_vocoder_checkpoint = "microsoft/speecht5_hifigan"
stt_checkpoint = "microsoft/speecht5_asr"
bark_checkpoint = "suno/bark-small"
bark_voice_presets="v2/en_speaker_0"
tacotron2_checkpoint = "speechbrain/tts-tacotron2-ljspeech"
hifigan_checkpoint = "speechbrain/tts-hifigan-ljspeech"
EMBEDDING_CHECKPOINT = "speechbrain/spkrec-xvect-voxceleb"
os.makedirs(os.path.join(CWD,"models","TTS","embeddings"),exist_ok=True)
TTS_MODELS_DIR = os.path.join(CWD,"models","TTS")
STT_MODELS_DIR = os.path.join(CWD,"models","STT")
DEFAULT_SPEAKER = os.path.join(TTS_MODELS_DIR,"embeddings","Sayano.npy")

def __speecht5__(text, speaker_embedding=None, device="cpu"):
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    tts_vocoder = SpeechT5HifiGan.from_pretrained(speecht5_vocoder_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_vocoder_checkpoint),device_map=device)
    tts_processor = SpeechT5Processor.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    inputs = tts_processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :tts_model.config.max_text_positions]

    dtype = torch.float32 if "cpu" in str(device) else torch.float16
    speech = tts_model.generate_speech(input_ids.to(device), speaker_embedding.to(device).to(dtype), vocoder=tts_vocoder)
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
    dtype = torch.float32 if "cpu" in str(device) else torch.float16
    bark_processor = AutoProcessor.from_pretrained(
        bark_checkpoint,
        cache_dir=os.path.join(TTS_MODELS_DIR,bark_checkpoint),
        torch_dtype=dtype)
    bark_model = BarkModel.from_pretrained(
        bark_checkpoint,
        cache_dir=os.path.join(TTS_MODELS_DIR,bark_checkpoint),
        torch_dtype=dtype).to(device)
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
    return remix_audio((speech, 22050),target_sr=16000,to_mono=True,norm=True)

def __edge__(text, speaker="en-US-JennyNeural"):
    import edge_tts
    import asyncio
    from threading import Thread
    temp_dir = os.path.join(BASE_CACHE_DIR,"tts","edge",speaker)
    os.makedirs(temp_dir,exist_ok=True)
    tempfile = os.path.join(temp_dir,f"{hashlib.md5(text.encode('utf-8')).hexdigest()}.wav")

    async def fetch_audio():
        communicate = edge_tts.Communicate(text, speaker)

        try:
            with open(tempfile, "wb") as data:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data.write(chunk["data"])
        except Exception as e:
            print(e)
    
    thread = Thread(target=asyncio.run, args=(fetch_audio(),),name="edge-tts",daemon=True)
    thread.start()
    thread.join()
    
    try:
        audio, sr = load_input_audio(tempfile,sr=16000)
        return audio, sr
    except Exception as e:
        print(e)
        return None

def __silero__(text, speaker="lj_16khz"):
    from silero import silero_tts
    
    model, symbols, sample_rate, _, apply_tts = silero_tts(
        repo_or_dir='snakers4/silero-models',
        language="en",
        speaker=speaker)

    audio = apply_tts(texts=[text],
                      model=model,
                      symbols=symbols,
                        sample_rate=sample_rate,
                        device="cpu")
    return audio[0].cpu().numpy(), 16000
    
def __vits__(text,speaker=os.path.join(CWD,"models","VITS","pretrained_ljs.pth")):
    from lib.infer_pack.models import SynthesizerTrn
    from lib.infer_pack.text.symbols import symbols
    from lib.infer_pack.text import text_to_sequence
    from lib.infer_pack.commons import intersperse
    from lib import utils

    hps = utils.get_hparams_from_file(os.path.join(CWD,"models","VITS","configs","ljs_base.json"))
    def get_text(text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint(speaker, net_g, None)

    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.678, noise_scale_w=0.6, length_scale=1.1)[0][0,0].data.cpu().float().numpy()
    return audio, hps.data.sampling_rate

def generate_speech(text, speaker=None, method="speecht5",device="cpu",dialog_only=False):
    
    text = english_cleaners(text.strip(),dialog_only=dialog_only) #clean text
    if text and len(text) == 0:
        return (np.zeros(0).astype(np.int16),16000)
    
    speaker_embedding = None
    
    if method=="speecht5":
        if type(speaker)==str:
            embedding_path = os.path.join(TTS_MODELS_DIR,"embeddings",f"{speaker}.npy")
            if os.path.isfile(embedding_path):
                speaker_embedding = np.load(embedding_path)
                speaker_embedding = torch.tensor(speaker_embedding).half()
            elif os.path.isfile(DEFAULT_SPEAKER):
                print(f"Speaker {speaker} not found, using default speaker...")
                speaker_embedding = np.load(DEFAULT_SPEAKER)
                speaker_embedding = torch.tensor(speaker_embedding).half()
            else: raise ValueError(f"Must provider a speaker_embedding for {method} inference!")
        else: speaker_embedding = speaker
        return __speecht5__(text,speaker_embedding,device)
    elif method=="bark":
        return __bark__(text,device)
    elif method=="tacotron2":
        return __tacotron2__(text,device)
    elif method=="edge":
        return __edge__(text)
    elif method=="vits":
        return __vits__(text)
    elif method=="silero":
        return __silero__(text)
    else: return None

def load_stt_models(method="vosk",recognizer=None):
    if method=="vosk":
        assert recognizer is not None, "Must provide recognizer object for vosk model"
        from vosk import Model
        import zipfile
        model_path = os.path.join(STT_MODELS_DIR,"vosk-model-en-us-0.22-lgraph")
        if not os.path.exists(model_path):
            temp_dir = os.path.join(BASE_CACHE_DIR,"zips")
            os.makedirs(temp_dir,exist_ok=True)
            name = os.path.basename(model_path)
            zip_path = os.path.join(temp_dir,name)+".zip"
            download_link = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"
            download_file((zip_path,download_link))
            print(f"extracting zip file: {zip_path}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(STT_MODELS_DIR)
            print(f"finished extracting zip file")

        model = Model(model_path=model_path,lang="en")
        recognizer.vosk_model = model
        return {
            "recognizer": recognizer,
            "model": model
        }
    elif method=="speecht5":
        from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
        processor = SpeechT5Processor.from_pretrained(stt_checkpoint,cache_dir=os.path.join(STT_MODELS_DIR,stt_checkpoint))
        generator = SpeechT5ForSpeechToText.from_pretrained(stt_checkpoint,cache_dir=os.path.join(STT_MODELS_DIR,stt_checkpoint))
        
        return {
            "processor": processor,
            "generator": generator
        }
    
def transcribe_speech(input_audio,stt_models=None,stt_method="vosk",denoise=False):

    if stt_models is None:
        stt_models = load_stt_models(stt_method)

    if stt_method=="vosk":
        recognizer = stt_models["recognizer"]
        model = stt_models["model"]
        recognizer
        input_data = recognizer.recognize_vosk(audio)
        input_data = json.loads(input_data)
        transcription = input_data["text"] if "text" in input_data else None
        return transcription
    elif stt_method=="speecht5":
        processor = stt_models["processor"]
        model = stt_models["generator"]
        audio, sr = input_audio
        slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500
        )
        transcription = ""

        for slice in slicer.slice(audio):
            # if denoise: audio = nr.red`uce_noise(audio,sr=sr)
            inputs = processor(audio=slice.T, sampling_rate=sr, return_tensors="pt")
            
            audio_len = int(len(slice)*6.25//sr)+1 #average 2.5 words/s spoken at 2.5 token/word

            predicted_ids = model.generate(**inputs, max_length=min(150,audio_len))
            print(predicted_ids)

            result = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            transcription += result[0]

        return transcription
    return None