import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from num2words import num2words
# import noisereduce as nr
from transformers import AutoProcessor, AutoModel
import scipy

num_channels = 2
sample_width = 2
frame_rate = 44100
num_frames = 0 
compression_type = 'NONE'
compression_name = 'not compressed'

tts_checkpoint = "microsoft/speecht5_tts"

speaker_embeddings = {
    "male": "./models/tts/embeddings/male.npy",
    "female": "./models/tts/embeddings/female.npy"
}

def speecht5(text='', speaker='male'):
    if len(text.strip()) == 0:
        return (16000, np.zeros(0).astype(np.int16))
    tts_processor = SpeechT5Processor.from_pretrained(tts_checkpoint,cache_dir="./models/tts")
    tts_model = SpeechT5ForTextToSpeech.from_pretrained(tts_checkpoint,cache_dir="./models/tts")
    tts_vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan",cache_dir="./models/tts")
    text = parseText(text) #convert numbers to words
    inputs = tts_processor(text=text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_ids = input_ids[..., :tts_model.config.max_text_positions]
    speaker_embedding = np.load(speaker_embeddings[speaker])
    speaker_embedding = torch.tensor(speaker_embedding).unsqueeze(0)
    speech = tts_model.generate_speech(input_ids, speaker_embedding, vocoder=tts_vocoder)
    speech = (speech.numpy() * 32767).astype(np.int16)
    # sf.write(file, speech, 16000, 'PCM_16')
    return speech, 16000

def bark(text,file='test.wav'):
    bark_processor = AutoProcessor.from_pretrained("suno/bark-small")
    bark_model = AutoModel.from_pretrained("suno/bark-small")

    inputs = bark_processor(
    text=[text],
    return_tensors="pt",
    )
    speech_values = bark_model.generate(**inputs, do_sample=True)
    sampling_rate = bark_model.generation_config.sample_rate
    scipy.io.wavfile.write(file, rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

def parseText(text):
    words = text.split()
    words = [num2words(word) if word.isdigit() else word for word in words]
    return ' '.join(words)