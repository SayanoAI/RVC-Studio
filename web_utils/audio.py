
import io
import numpy as np
import librosa
import soundfile as sf

MAX_INT16 = 32768
SUPPORTED_AUDIO = ["wav","mp3","flac","ogg"]

def remix_audio(input_audio,target_sr=None,norm=False,to_int16=False,resample=False,to_mono=False,axis=0,**kwargs):
    audio = np.array(input_audio[0],dtype="float32")
    if target_sr is None: target_sr=input_audio[1]

    print(f"before remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()} sr={input_audio[1]}")
    if resample or input_audio[1]!=target_sr:
        audio = librosa.core.resample(np.array(input_audio[0],dtype="float32"),orig_sr=input_audio[1],target_sr=target_sr,**kwargs)
    
    if to_mono and audio.ndim>1: audio=np.nanmedian(audio,axis)

    if norm: audio = librosa.util.normalize(audio)

    audio_max = np.abs(audio).max() / 0.95
    if audio_max > 1: audio /= audio_max
    
    if to_int16: audio = np.clip(audio * MAX_INT16, a_min=-MAX_INT16+1, a_max=MAX_INT16-1).astype("int16")
    print(f"after remix: shape={audio.shape}, max={audio.max()}, min={audio.min()}, mean={audio.mean()}, sr={target_sr}")

    return audio, target_sr

def load_input_audio(fname,sr=None,**kwargs):
    sound = librosa.load(fname,sr=sr,**kwargs)
    print(f"loading sound {fname} {sound[0].shape} {sound[1]}")
    return sound
   
def save_input_audio(fname,input_audio,sr=None,to_int16=False):
    print(f"saving sound to {fname}")
    audio=np.array(input_audio[0],dtype="float32")
    if to_int16:
        max_a = np.abs(audio).max() * .99
        if max_a<1:
            audio=(audio*max_a*MAX_INT16)
        audio=audio.astype("int16")
    try:        
        sf.write(fname, audio, sr if sr else input_audio[1])
        return True
    except:
        return False
    
def audio_to_bytes(audio,sr,format='WAV'):
    bytes_io = io.BytesIO()
    sf.write(bytes_io, audio, sr, format=format)
    return bytes_io.read()

def bytes_to_audio(data):
    bytes_io = io.BytesIO(data)
    audio = sf.read(bytes_io)
    return audio