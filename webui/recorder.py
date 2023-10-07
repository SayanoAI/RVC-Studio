import pydub
import pyaudio
import wave
import threading

from webui.audio import audio_to_bytes, bytes_to_audio

# Define some constants
CHUNK = 1024 # Number of frames per buffer
FORMAT = pyaudio.paInt16 # Audio format
CHANNELS = 2 # Number of channels
RATE = 44100 # Sampling rate in Hz
RECORD_SECONDS = 10 # Duration of recording in seconds
WAVE_FILE = "output.wav" # Output file name

# Define a class that can record and play audio chunks in real time
class RecorderPlayback:
    def __init__(self):
        # Initialize the PyAudio object
        self.p = pyaudio.PyAudio()

        # Create a stream for recording
        self.record_stream = self.p.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=CHUNK)

        # Create a stream for playback
        self.play_stream = self.p.open(format=FORMAT,
                                       channels=CHANNELS,
                                       rate=RATE,
                                       output=True,
                                       frames_per_buffer=CHUNK)

        # Create a list to store the audio chunks
        self.frames = []

        # Create a flag to indicate if the recording is active
        self.recording = False

    def start(self):
        # Set the flag to True
        self.recording = True

        # Create a thread for recording
        self.record_thread = threading.Thread(target=self.record,daemon=True)
        self.record_thread.start()

        # Create a thread for playback
        self.play_thread = threading.Thread(target=self.play,daemon=True)
        self.play_thread.start()

    def stop(self):
        # Set the flag to False
        self.recording = False

        # Wait for the threads to finish
        self.record_thread.join()
        self.play_thread.join()

        # Close the streams
        self.record_stream.stop_stream()
        self.record_stream.close()
        self.play_stream.stop_stream()
        self.play_stream.close()

        # Terminate the PyAudio object
        self.p.terminate()

    def record(self):
        # Loop until the flag is False
        while self.recording:
            # Read a chunk of data from the input device
            data = self.record_stream.read(CHUNK)

            # Process the data (for example, apply some filter or effect)
            data = self.process(data)

            # Append the data to the list of frames
            self.frames.append(data)

    def play(self):
        # Loop until the flag is False
        while self.recording:
            # Check if there are any frames in the list
            if len(self.frames) > 0:
                # Pop the first frame from the list
                data = self.frames.pop(0)

                # Write the data to the output device
                self.play_stream.write(data)

    def process(self, data):
        # This is a dummy function that does nothing to the data
        # You can implement your own logic here to modify the data as you wish
        sound = pydub.AudioSegment(
                            data=data,
                            sample_width=FORMAT,
                            frame_rate=RATE,
                            channels=len(CHANNELS),
                        )
        # audio = bytes_to_audio(data)
        data = audio_to_bytes((sound.get_array_of_samples(),RATE))
        return data

    def save(self):
        # Create a wave file object
        wf = wave.open(WAVE_FILE, 'wb')

        # Set the parameters of the file
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        # Write the frames to the file
        wf.writeframes(b''.join(self.frames))

        # Close the file
        wf.close()

# Create an instance of the class
rp = RecorderPlayback()

# Start recording and playback
rp.start()

# Wait for some time (for example, RECORD_SECONDS)
# time.sleep(RECORD_SECONDS)

# # Stop recording and playback
# rp.stop()

# # Save the recorded audio to a file
# rp.save()


# import asyncio
# from collections import deque
# import threading, webrtcvad
# import numpy as np
# import time
# import pyaudio, pydub
# import noisereduce
# from vc_infer_pipeline import get_vc, vc_single
# from webui.utils import gc_collect


# class PlaybackRecorder:
#     # Initialize the character with a name and a voice
#     def __init__(self, input_device = None, output_device = None, vad_mode=2,**kwargs):
#         self.input_device = input_device
#         self.output_device = output_device
#         self.sample_rate = 16000 # same rate as hubert model
#         self.chunk_size = 1024
#         self.is_recording = False
#         self.recognizer = None
#         self.recorder = None
#         self.frames_deque_lock = threading.Lock()
#         self.frames_deque = deque([])
#         self.vad = webrtcvad.Vad(vad_mode)
#         self.rvc_models = None
#         self.args = kwargs

#     def __del__(self):
#         self.stop_listening()
#         del rvc_models
#         gc_collect()

#     def load_rvc_models(self):
#         if self.rvc_model is None:
#             self.rvc_model = get_vc(self.model_name,config=self.args["config"],device=self.args["device"])
#         return self.rvc_model

#     # Define a method to run the STT and TTS in the background and be non-blocking
#     async def start_recording(self):
#         import speech_recognition as sr
        
#         # Create a speech recognizer instance
#         self.recognizer = sr.Recognizer()
#         # self.recognizer.energy_threshold = 2000
#         # self.recognizer.pause_threshold = 2.
#         self.is_recording = True
        
#         # Start listening to the microphone in the background and call the callback function when speech is detected
#         self.autoplay = False
        
#         # Create a microphone instance
#         # Start listening to mic
#         self.recorder = threading.Thread(target=asyncio.run,args=(self.microphone_callback(),),daemon=True,name="recorder")
#         self.recorder.start()
       
#     # Define a callback function that will be called when speech is detected
#     def microphone_callback(self):
#         import speech_recognition as sr
#         # with sr.Microphone(0,sample_rate=self.sample_rate) as source, self.lock:
#         #     self.recognizer.adjust_for_ambient_noise(source, duration=self.recognizer.pause_threshold)
#         recognizer = sr.Recognizer()

#         while self.is_recording:

#             with sr.Microphone(sample_rate=self.sample_rate,device_index=self.input_device,chunk_size=self.chunk_size) as source:
#                 try:
#                     audio = recognizer.listen(source)

#                     if not self.is_recording:
#                         return self.stop_listening()
                    
#                     with self.frames_deque_lock:
#                         if self.vad.is_speech(audio.get_raw_data(),audio.sample_rate,length=audio.sample_width):
#                             self.frames_deque.append(audio)

#                     # if len(audio.frame_data)>self.sample_rate*2:
#                     # input_audio = bytes_to_audio(audio.get_wav_data())
#                     # self.text = transcribe_speech(input_audio,stt_models=self.stt_models,stt_method=self.stt_method,denoise=True)
#                 except Exception as e:
#                     print(e)
    
#     async def process_frames(self):
#         stream = None
#         p = pyaudio.PyAudio()

#         while True:
#             if stream is None:
#                 stream = p.open(
#                     format=pyaudio.paInt16,
#                     channels=1,
#                     rate=self.sample_rate,
#                     output=True,
#                     output_device_index=self.output_device
#                 )
#                 stream.start_stream()
#                 print("Stream started")

#             if self.is_recording:
#                 sound_chunk = pydub.AudioSegment.empty()
#                 print("Running. Say something!")

#                 audio_frames = []
#                 with self.frames_deque_lock:
#                     while len(self.frames_deque) > 0:
#                         frame = self.frames_deque.popleft()
#                         audio_frames.append(frame)

#                     if len(audio_frames) == 0:
#                         time.sleep(0.1)
#                         print("No frame arrived.")
#                         continue

#                 for audio_frame in audio_frames:
#                     if audio_frame:
#                         sound = pydub.AudioSegment(
#                             data=audio_frame.to_ndarray().tobytes(),
#                             sample_width=audio_frame.format.bytes,
#                             frame_rate=audio_frame.sample_rate,
#                             channels=len(audio_frame.layout.channels),
#                         )
#                         sound_chunk += sound

#                 if len(sound_chunk) > 0:
#                     sound_chunk = sound_chunk.set_channels(1).set_frame_rate(16000)
#                     # print(len(sound_chunk),sound_chunk.frame_rate,sound_chunk.sample_width)
#                     sound_chunk = sound_chunk.get_array_of_samples()
#                     if len(sound_chunk) > 2048:
#                         audio = noisereduce.reduce_noise(y=sound_chunk, sr=16000)
                        
#                         # Apply Spectral Gate to noisy speech signal
#                         # tg = TorchGate(sr=16000, nonstationary=True).to(state.device)
#                         # noisy_speech = torch.tensor(sound_chunk,device=state.device).unsqueeze(0)
#                         # audio = tg(noisy_speech).cpu().numpy()
                        
#                         changed_voice = vc_single(
#                             input_audio=(audio,16000),
#                             **self.rvc_options,
#                             **self.rvc_models
#                         )
#                         if changed_voice is not None:
#                             dtype = "int16" if np.abs(changed_voice[0]).max()>1 else "float32" 
#                             data = (changed_voice[0]).astype(dtype)
#                             stream.write(data.tobytes())
#                             audio_frames = []
#             else:
#                 print("Stopped.")
#                 stream.stop_stream()
#                 stream.close()
#                 break

#     def stop_listening(self):
#         if self.recorder:
#             print("stopped listening to mic...")
#             self.is_recording = False
#             self.recorder.join(1)
#             self.recorder = self.recognizer = None