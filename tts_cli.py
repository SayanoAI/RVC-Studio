import numpy as np
import torch
import os

from lib.infer_pack.text.cleaners import english_cleaners

from web_utils.audio import MAX_INT16, load_input_audio, remix_audio

CWD = os.getcwd()
speecht5_checkpoint = "microsoft/speecht5_tts"
speecht5_vocoder_checkpoint = "microsoft/speecht5_hifigan"
stt_checkpoint = "microsoft/speecht5_asr"
bark_checkpoint = "suno/bark-small"
bark_voice_presets="v2/en_speaker_0"
tacotron2_checkpoint = "speechbrain/tts-tacotron2-ljspeech"
hifigan_checkpoint = "speechbrain/tts-hifigan-ljspeech"
EMBEDDING_CHECKPOINT = "speechbrain/spkrec-xvect-voxceleb"
os.makedirs(os.path.join(CWD,"models","tts","embeddings"),exist_ok=True)
TTS_MODELS_DIR = os.path.join(CWD,"models","tts")
DEFAULT_SPEAKER = os.path.join(TTS_MODELS_DIR,"embeddings","Sayano.npy")

def __speecht5__(text, speaker_embedding=None, device="cpu"):
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    tts_vocoder = SpeechT5HifiGan.from_pretrained(speecht5_vocoder_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_vocoder_checkpoint),device_map=device)
    tts_processor = SpeechT5Processor.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(speecht5_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,speecht5_checkpoint),device_map=device)
    inputs = tts_processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :tts_model.config.max_text_positions]

    dtype = torch.float32 if "cpu" in device else torch.float16
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
    dtype = torch.float32 if "cpu" in device else torch.float16
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

    async def fetch_audio():
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        communicate = edge_tts.Communicate(text, speaker)
        tempfile = os.path.join("output","edge_tts.wav")
        with open(tempfile, "wb") as data:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    data.write(chunk["data"])
                # elif chunk["type"] == "WordBoundary":
                #     print(f"WordBoundary: {chunk}")
        
        return load_input_audio(tempfile,sr=16000)
    
    audio, sr = asyncio.run(fetch_audio())
    # audio = np.frombuffer(stream.getbuffer())
    # print(audio.shape,audio.max(),audio.min(),audio.mean(),sr)
    # return as numpy array
    return audio, sr

def __vits__(text,speaker="./models/VITS/pretrained_ljs.pth"):
    from lib.infer_pack.models import SynthesizerTrn
    from lib.infer_pack.text.symbols import symbols
    from lib.infer_pack.text import text_to_sequence
    from lib.infer_pack.commons import intersperse
    from lib.train import utils

    hps = utils.get_hparams_from_file("./models/VITS/configs/ljs_base.json")
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

def generate_speech(text, speaker=None, method="speecht5",device="cpu"):
    
    text = english_cleaners(text.strip()) #clean text
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
    else: return None

def load_stt_models():
    from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
    processor = SpeechT5Processor.from_pretrained(stt_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,stt_checkpoint))
    generator = SpeechT5ForSpeechToText.from_pretrained(stt_checkpoint,cache_dir=os.path.join(TTS_MODELS_DIR,stt_checkpoint))
    
    return {
        "processor": processor,
        "generator": generator
    }
    
def transcribe_speech(input_audio,stt_models=None):
    if stt_models is None:
        stt_models = load_stt_models()

    processor = stt_models["processor"]
    model = stt_models["generator"]

    inputs = processor(audio=input_audio[0].T, sampling_rate=input_audio[1], return_tensors="pt")
    
    audio_len = int(len(input_audio[0])*6.25//input_audio[1])+1 #average 2.5 words/s spoken at 2.5 token/word

    predicted_ids = model.generate(**inputs, max_length=min(150,audio_len))

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]