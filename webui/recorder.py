from collections import deque
import numpy as np
import pyaudio
import threading
import torch
from webrtcvad import Vad
from rvc_for_realtime import RVC

from webui.utils import gc_collect

# Define a class that can record and play audio chunks in real time
class RecorderPlayback:
    def __init__(self, agg=0, chunk=160,
                 channels=1, sr=16000, max_delay=200, silence_threshold=0.01):
        # Initialize the PyAudio object
        self.p = pyaudio.PyAudio()
        self.vad = Vad(agg)
        self.chunk = chunk
        self.input_format = pyaudio.paFloat32
        self.output_format = pyaudio.paFloat32
        self.channels = channels
        self.sr = sr
        self.tgt_sr = sr
        self.silence_threshold = silence_threshold
        self.rvc_model = None
        self.voice_model = None
        self.rvc_options = {}
        self.rvc_model_lock = threading.Lock()

        # stream for recording and playback
        self.record_stream = self.play_stream = None

        # Create a list to store the audio chunks
        self.frames = deque([])
        self.max_frames = int(max_delay/1000*self.sr/self.chunk) #200 ms delay

        # Create a flag to indicate if the recording is active
        self.recording = False

    def start(self, voice_model, config, device,
              input_device_index = None, output_device_index = None,
              **options):
        torch.cuda.empty_cache()
        self.config = config
        self.device = device
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        self.rvc_options.update(options)
        self.voice_model = voice_model
        
        # Set the flag to True
        self.recording = True

        # Create a thread for recording
        self.record_thread = threading.Thread(target=self.record,daemon=True)

        if not self.record_thread.is_alive():
            self.record_thread.start()

    def stop(self):

        # Set the flag to False
        self.recording = False

        # Wait for the threads to finish
        self.record_thread.join(1)
        self.record_thread = None
        # self.play_thread.join(1)

        # reset stream
        # self.record_stream = self.play_stream = None

    def __del__(self):
        # Terminate the PyAudio object
        self.p.terminate()
        del self.rvc_model
        gc_collect()

    def __repr__(self):
        return f"{self.__class__}(voice_model={self.voice_model},recording={self.recording})"

    def record(self):

        with self.rvc_model_lock:
            if self.rvc_model is None:
                self.rvc_model = self.load_rvc_model(self.voice_model, self.config, self.device)
            
                # Create a stream for recording
                self.record_stream = self.p.open(format=self.input_format,
                                                channels=self.channels,
                                                rate=self.sr,
                                                input=True,
                                                input_device_index=self.input_device_index,
                                                frames_per_buffer=self.chunk)
                print("started listening")
                self.play_stream = self.p.open(format=self.input_format,
                                                channels=self.channels,
                                                rate=self.rvc_model.tgt_sr,
                                                output=True,
                                                output_device_index=self.output_device_index,
                                                frames_per_buffer=self.chunk)
                print("started playing")

        # Loop until the flag is False
        while self.recording:
            if len(self.frames)<self.sr/self.chunk:
                # Read a chunk of data from the input device
                data = self.record_stream.read(self.sr,exception_on_overflow = False)
                # Process the data (for example, apply some filter or effect)
                audio =  np.frombuffer(data, dtype=np.float32)
                if np.std(audio)>self.silence_threshold:
                    # Append the data to the list of frames
                    audio = self.rvc_model.vc(audio,**self.rvc_options)
                    self.frames.append(audio)
            if len(self.frames)>0:
                # audio = np.concatenate(self.frames)
                audio = self.frames.popleft()
                # audio = self.rvc_model.vc(audio,**self.rvc_options)
                # audio = remix_audio((audio,self.sr),target_sr=self.rvc_model.tgt_sr,norm=True)[0]
                self.play_stream.write(audio.tobytes())
                # self.frames.clear()
                # time.sleep(self.chunk/self.sr)
        
        # Close the streams
        if self.record_stream.is_active():
            self.record_stream.stop_stream()
            self.record_stream.close()
            self.record_stream = None
            print("stopped listening")
        if self.play_stream and self.play_stream.is_active():
            self.play_stream.stop_stream()
            self.play_stream.close()
            self.play_stream = None
            print("stopped playing")

    # def play(self):
    #     with self.rvc_model_lock:
    #         if self.rvc_model:
    #             # Create a stream for playback
    #             self.play_stream = self.p.open(format=self.output_format,
    #                                         channels=self.channels,
    #                                         rate=self.rvc_model.sr,
    #                                         output=True,
    #                                         output_device_index=self.output_device_index,
    #                                         frames_per_buffer=self.chunk)
    #             print("start playing")
        
    #     # Loop until the flag is False
    #     while self.recording:
    #         # Check if there are any frames in the list
    #         if len(self.frames) > 0 and self.play_stream is not None:
    #             # Pop the first frame from the list
    #             audio = self.frames.popleft()
    #             # audio = self.rvc_model.process_output(*data)
    #             # print(audio.max(),audio.min(),audio.shape)
                
    #             # Write the data to the output device
    #             self.play_stream.write(audio.tobytes())

    #     # Close the streams
    #     if self.play_stream and self.play_stream.is_active():
    #         self.play_stream.stop_stream()
    #         self.play_stream.close()
    #         print("stopped playing")
    
    def load_rvc_model(self,model_name,config,device):
        print(f"loading {model_name}...")
        if self.rvc_model is None or self.rvc_model.model_name!=model_name:
            if self.rvc_model: del self.rvc_model
            self.rvc_model = RVC(model_name,config=config,device=device)
            gc_collect()
        print(f"{model_name} finished loading")
        return self.rvc_model