import time
import numpy as np
import pyaudio
import threading
import torch
import webrtcvad
from rvc_for_realtime import RVC
from webui.audio import remix_audio

from webui.utils import ObjectNamespace, gc_collect

# Define a class that can record and play audio chunks in real time
class RecorderPlayback:
    def __init__(self, agg=0, chunk=160, channels=1, sr=16000, silence_threshold=0.01):
        # Initialize the PyAudio object
        self.p = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(agg)
        self.chunk = chunk
        self.format = pyaudio.paFloat32
        self.channels = channels
        self.sr = sr
        self.tgt_sr = sr
        self.silence_threshold = silence_threshold
        self.rvc_model = None
        self.voice_model = None
        self.rvc_options = ObjectNamespace()
        self.lock = threading.Lock()

        # Create a flag to indicate if the recording is active
        self.recording = False

    def update_options(self, options):
        with self.lock:
            self.rvc_options.update(options)

    def start(self, voice_model, config, device,
              input_device_index = None, output_device_index = None,
              **options):
        torch.cuda.empty_cache()
        self.config = config
        self.device = device
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.update_options(options)
        self.voice_model = voice_model
        self.rvc_model = self.load_rvc_model(self.voice_model, self.config, self.device)
        self.tgt_sr = self.rvc_model.tgt_sr
        
        # Set the flag to True
        self.recording = True

        # Create a thread for recording
        self.record_thread = threading.Thread(target=self.record,daemon=False)

        if not self.record_thread.is_alive():
            self.record_thread.start()

    def stop(self):
        # Set the flag to False
        self.recording = False

        # Wait for the threads to finish
        self.record_thread.join(1)
        self.record_thread = None

    def __del__(self):
        # Terminate the PyAudio object
        self.p.terminate()
        del self.rvc_model
        gc_collect()

    def __repr__(self):
        return f"{self.__class__}(voice_model={self.voice_model},recording={self.recording})"

    def record(self):
        print("started listening")
        self.io_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.tgt_sr,
            input=True,
            output=True,
            input_device_index=self.input_device_index,
            output_device_index=self.output_device_index,
            stream_callback=self.process_audio,
            frames_per_buffer=self.tgt_sr
        )
        self.io_stream.start_stream()
        # Loop until the flag is False
        with self.lock:
            while self.recording:
                time.sleep(1.)
        print("stopped listening")
        self.io_stream.stop_stream()
        self.io_stream.close()
    
    def process_audio(self, data, frame_count, *args, **kwargs):
        # Process the data (for example, apply some filter or effect)
        audio =  np.frombuffer(data, dtype=np.float32)

        if np.std(audio)>self.silence_threshold:
            audio = remix_audio((audio,self.tgt_sr),target_sr=self.sr)[0]

            # if self.is_speech(audio):            
            audio = self.rvc_model.vc(audio,**self.rvc_options)

            if len(audio)<frame_count:
                audio = np.pad(audio,(0,frame_count),mode="linear_ramp",end_values=0)
        else: audio = np.zeros(frame_count)
            
        return (audio, pyaudio.paContinue)
    
    def load_rvc_model(self,model_name,config,device):
        print(f"loading {model_name}...")
        if self.rvc_model is None or self.rvc_model.model_name!=model_name:
            if self.rvc_model: del self.rvc_model
            self.rvc_model = RVC(model_name,config=config,device=device)
            gc_collect()
        print(f"{model_name} finished loading")
        return self.rvc_model
    
    def is_speech(self, audio):
        length = int(self.sr * .03)
        slices = int(len(audio)/length)+1
        
        for segment in np.array_split(audio,slices):
            if self.vad.is_speech(segment.tobytes(),self.sr,length=length): return True
        return False