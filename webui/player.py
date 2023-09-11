import multiprocessing as mp
import os
import random
import threading
from typing import Iterable
from uvr5_cli import split_audio
import asyncio
from vc_infer_pipeline import get_vc, vc_single
from webui.audio import load_input_audio, save_input_audio, merge_audio
from webui.downloader import BASE_CACHE_DIR
import sounddevice as sd

def convert_song(
    audio_path, # song name
    rvc_models, # RVC model
    uvr5_name=[], # UVR5 models
    preprocess_model=None, # reverb removal model
    device="cuda",
    agg=10,
    merge_type="median",
    use_cache=True,

    # voice change
    f0_up_key=0,
    f0_method="rmvpe",
    index_rate=.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=.2,
    protect=0.2,

    **kwargs
):
    cache_dir = os.path.join(BASE_CACHE_DIR,"playlist",rvc_models["model_name"])
    os.makedirs(cache_dir,exist_ok=True)
    song_path = os.path.join(cache_dir,os.path.basename(audio_path).split(".")+".mp3")
    if os.path.isfile(song_path):
        return load_input_audio(song_path)
    
    print(f"unused args: {kwargs}")
    input_vocals, input_instrumental, input_audio = split_audio(
        uvr5_name,
        audio_path=audio_path,
        preprocess_model=preprocess_model,
        device=device,
        agg=agg,
        use_cache=use_cache,
        merge_type=merge_type
        )
    changed_vocals = vc_single(
        input_audio=input_vocals,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        **rvc_models
    )

    mixed_audio = merge_audio(changed_vocals,input_instrumental,sr=input_audio[1])

    if use_cache:
        save_input_audio(song_path,mixed_audio)

    return mixed_audio

class PlaylistPlayer:
    def __init__(self, playlist: Iterable[str], model_name, config, volume=1.0, shuffle=False, loop=False, **args):

        # playlist is a list of song filenames
        self.playlist = playlist
        self.index = 0 # current song index
        self.paused = False # pause flag
        self.stopped = False # stop flag
        self.lock = mp.Lock() # lock for synchronization
        self.queue = mp.Queue(3) # queue for processed songs
        self.model_name = model_name
        self.config = config
        self.args = args
        self.current_song = None
        self.loop = loop
        self.volume = volume
        
        if shuffle: self.shuffle()

        self.thread1 = threading.Thread(target=asyncio.run,args=(self.play_song(),),name="play_song")
        self.thread1.start()
        self.thread2 = threading.Thread(target=asyncio.run,args=(self.process_song(),),name="process_song") # process for converting songs
        self.thread2.start()

    def __repr__(self):
        status = "paused" if self.paused else f"playing: {self.current_song}"
        return f"PlaylistPlayer[{self.index+1}/{len(self.playlist)}] {self.thread1} {self.thread2} {status} ({self.queue.qsize()} in queue)"
    
    def __del__(self):
        sd.stop()
        self.stop()
        self.thread2.join(1) #stop processing
        self.thread1.join(1) #sto playing
    
    def set_args(self, **args):
        # update arguments
        with self.lock:
            self.args = args

    def load_model(self):
        return get_vc(self.model_name,config=self.config,device=self.args["device"])

    async def play_song(self):
        
        item = None
        # play the songs from the queue
        while not self.stopped:
            with self.lock:
                # sd.wait()
                if item is None and not self.queue.empty():
                    # get the next song data and sample rate from the queue
                    item = self.queue.get(block=False)

                if item and not self.stopped and not self.paused:
                    self.current_song, input_audio = item
                    data, fs = input_audio
                    sd.wait() # wait for old song to finish
                    sd.play((data*self.volume).astype("int16"),samplerate=fs)
                    item=None

    async def process_song(self):
        
        # convert the songs in the playlist and put them in the queue
        rvc_models = self.load_model()
        
        while not self.stopped and self.index < len(self.playlist):
            with self.lock:
                if not self.queue.full() and not self.paused:
                    # get the next song filename from the playlist
                    song = self.playlist[self.index]

                    try:
                        # call the convert_song function on it (replace with your own function)
                        input_audio = convert_song(song,rvc_models,**self.args)
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
        with self.lock:
            random.shuffle(self.playlist)

    def set_volume(self, volume):
        # shuffle the playlist order
        with self.lock:
            self.volume = volume

    def toggle_loop(self):
        with self.lock:
            self.loop=not self.loop

    def skip(self):
        
        # skip the current song and play the next one
        with self.lock:
            sd.stop()
            # increment the index
            self.index += 1
            # wrap around the index if it reaches the end of the playlist
            if self.index == len(self.playlist) and self.loop:
                self.index = 0

    def pause(self):
        
        # pause the playback
        with self.lock:
            sd.stop()
            self.paused = True

    def resume(self):
        # resume the playback
        with self.lock:
            self.paused = False

    def stop(self):
        # stop the playback and terminate the threads
        with self.lock:
            sd.stop()

            self.stopped = True
            while self.queue.qsize()>0:
                self.queue.get()