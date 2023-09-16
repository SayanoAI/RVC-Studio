import os
import queue
import random
import threading
from typing import Iterable

import numpy as np
from uvr5_cli import split_audio
import asyncio
from vc_infer_pipeline import get_vc, vc_single
from webui.audio import load_input_audio, save_input_audio, merge_audio
from webui.downloader import BASE_CACHE_DIR
import pyaudio

from webui.utils import gc_collect

def convert_song(
    audio_path, # song name
    rvc_models, # RVC model
    # uvr5_name=[], # UVR5 models
    # preprocess_model=None, # reverb removal model
    # device="cuda",
    # agg=10,
    # merge_type="median",
    split_audio_params,

    # voice change
    # f0_up_key=0,
    # f0_method="rmvpe",
    # index_rate=.75,
    # filter_radius=3,
    # resample_sr=0,
    # rms_mix_rate=.2,
    # protect=0.2,
    vc_single_params,

    use_cache=True,

    **kwargs
):
    cache_dir = os.path.join(BASE_CACHE_DIR,"playlist",rvc_models["model_name"])
    os.makedirs(cache_dir,exist_ok=True)
    song_path = os.path.join(cache_dir,os.path.basename(audio_path).split(".")[0]+".mp3")
    if os.path.isfile(song_path):
        return load_input_audio(song_path)
    
    print(f"unused args: {kwargs}")
    input_vocals, input_instrumental, input_audio = split_audio(
        audio_path=audio_path,
        **split_audio_params
        )
    changed_vocals = vc_single(
        input_audio=input_vocals,
        **vc_single_params,
        **rvc_models
    )

    mixed_audio = merge_audio(changed_vocals,input_instrumental,sr=input_audio[1])

    if use_cache:
        save_input_audio(song_path,mixed_audio)

    return mixed_audio

class PlaylistPlayer:
    def __init__(self, playlist: Iterable[str], model_name, config, volume=1.0, shuffle=False, loop=False, use_cache=True, **args):

        # playlist is a list of song filenames
        self.playlist = playlist
        self.index = 0 # current song index
        self.paused = False # pause flag
        self.stopped = False # stop flag
        self.lock = threading.Lock() # lock for synchronization
        self.queue = queue.Queue(1) # queue for processed songs
        self.model_name = model_name
        self.config = config
        self.args = args
        self.current_song = None
        self.loop = loop
        self.volume = volume
        self.CHUNKSIZE = 1024
        self.stream = None
        self.rvc_model = None
        self.use_cache = use_cache
        self.shuffled = False
        
        if shuffle: self.shuffle()

        self.thread1 = threading.Thread(target=asyncio.run,args=(self.play_song(),),name="play_song",daemon=True)
        self.thread1.start()
        self.thread2 = threading.Thread(target=asyncio.run,args=(self.process_song(),),name="process_song",daemon=True) # process for converting songs
        self.thread2.start()

    def __repr__(self):
        status = "paused" if self.paused else f"{'looping' if self.loop else 'playing'}: {self.current_song}"
        return f"PlaylistPlayer[{self.index+1}/{len(self.playlist)}] {self.thread1} {self.thread2} {status} ({self.queue.qsize()} in queue)"
    
    def __del__(self):
        try:
            self.thread2.join(1.)
        except Exception as e:
            print(f"failed to close thread2: {e}")
        try:
            self.thread1.join(1.)
        except Exception as e:
            print(f"failed to close thread1: {e}")
        try:
            self.stop()
            if self.stream: self.stream.close()
        except Exception as e:
            print(f"failed to close stream: {e}")
        del self.rvc_model
        gc_collect()
    
    def set_args(self, **args):
        # update arguments
        self.args.update(args)

    def load_model(self):
        if self.rvc_model is None:
            self.rvc_model = get_vc(self.model_name,config=self.config,device=self.args["device"])
        return self.rvc_model

    async def play_song(self):
        # initialize portaudio
        p = pyaudio.PyAudio()
        item = None
        # play the songs from the queue
        while not self.stopped:
            with self.lock:
                if item is None:
                    if self.queue.qsize()>0:
                        # get the next song data and sample rate from the queue
                        item = self.queue.get(block=False)
                    elif self.index == len(self.playlist) and not self.loop:
                        self.stop()
                elif not (self.stopped or self.paused):
                    self.current_song, input_audio = item
                    audio, sr = input_audio
                    format = pyaudio.paInt16 if np.abs(audio).max()>1 else pyaudio.paFloat32
                    dtype = "int16" if np.abs(audio).max()>1 else "float32" 
                    try:
                        self.stream = p.open(format=format, channels=1, rate=sr, output=True)
                        self.stream.start_stream()
                    
                        for i in range(0,len(audio),self.CHUNKSIZE):
                            if not (self.paused or self.stopped):
                                data = (audio[i:i+self.CHUNKSIZE]*self.volume).astype(dtype)
                                if self.stream.is_stopped(): break
                                self.stream.write(data.tostring())
                            else:
                                if self.stopped:
                                    break
                                while self.paused and not self.stopped:
                                    await asyncio.sleep(self.CHUNKSIZE/sr)
                        self.stream.stop_stream()
                        self.stream.close()
                    except Exception as e:
                        print(f"failed to stream {self.current_song}: {e}")
                    finally: item=None

        p.terminate()

    async def process_song(self):
        
        # convert the songs in the playlist and put them in the queue
        while not self.stopped and self.index < len(self.playlist):
            with self.lock:
                rvc_models = self.load_model()

                if not self.queue.full():
                    # get the next song filename from the playlist
                    song = self.playlist[self.index]

                    try:
                        # call the convert_song function on it (replace with your own function)
                        input_audio = convert_song(song,rvc_models,use_cache=self.use_cache,**self.args)
                        # put the song data and sample rate in the queue
                        self.queue.put((song, input_audio))
                    except Exception as e:
                        print(e)
                    await asyncio.sleep(1)
                    # increment the index
                    self.index += 1
                    # wrap around the index if it reaches the end of the playlist
                    if self.index == len(self.playlist) and self.loop:
                        self.index = 0

    def shuffle(self):
        # shuffle the playlist order
        not_played = self.playlist[self.index:]
        random.shuffle(not_played)
        self.playlist = self.playlist[:self.index] + not_played
        self.shuffled =True

    def set_volume(self, volume):
        # shuffle the playlist order
        self.volume = volume

    def set_loop(self, loop):
        self.loop = loop

    def skip(self):
        # skip the current song and play the next one
        if self.stream: self.stream.stop_stream()

    def pause(self):
        # pause the playback
        self.paused = True
        if self.stream: self.stream.stop_stream()

    def resume(self):
        # resume the playback
        self.paused = False
        if self.stream: self.stream.start_stream()

    def stop(self):
        # stop the playback and terminate the threads
        self.stopped = True
        while self.queue.qsize()>0:
            self.queue.get()
        if self.stream: self.stream.stop_stream()