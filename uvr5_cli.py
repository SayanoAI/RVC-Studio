import argparse
import os, torch, warnings

from lib.separators import MDXNet, UVR5Base, UVR5New
from webui import get_cwd
from lib.audio import load_input_audio, pad_audio, remix_audio, save_input_audio
from webui.downloader import BASE_CACHE_DIR
from webui.utils import gc_collect, get_optimal_threads

CWD = get_cwd()
CACHED_SONGS_DIR = os.path.join(BASE_CACHE_DIR,"songs")

warnings.filterwarnings("ignore")
import numpy as np

class Separator:
    def __init__(self, model_path, use_cache=False, device="cpu", cache_dir=None, **kwargs):
        dereverb = "reverb" in model_path.lower()
        deecho = "echo"  in model_path.lower()
        denoise = dereverb or deecho

        if "MDX" in model_path:
            self.model = MDXNet(model_path=model_path,denoise=denoise,device=device,**kwargs)
        elif "UVR" in model_path:
            self.model = UVR5New(model_path=model_path,device=device,dereverb=dereverb,**kwargs) if denoise else UVR5Base(model_path=model_path,device=device,**kwargs)
            
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.args = kwargs
    
    # cleanup memory
    def __del__(self):
        gc_collect()

    def run_inference(self, audio_path, format="mp3"):
        song_name = get_filename(os.path.basename(self.model_path).split(".")[0],**self.args) + f".{format}"
        
        # handles loading of previous processed data
        music_dir = os.path.join(
            os.path.dirname(audio_path) if self.cache_dir is None else self.cache_dir,
            os.path.basename(audio_path).split(".")[0])
        vocals_path = os.path.join(music_dir,".vocals")
        instrumental_path = os.path.join(music_dir,".instrumental")
        vocals_file = os.path.join(vocals_path,song_name)
        instrumental_file = os.path.join(instrumental_path,song_name)

        if os.path.isfile(instrumental_file) and os.path.isfile(vocals_file):
            vocals = load_input_audio(vocals_file,mono=True)
            instrumental = load_input_audio(instrumental_file,mono=True)
            input_audio = load_input_audio(audio_path,mono=True)
            return vocals, instrumental, input_audio
        
        return_dict = self.model.run_inference(audio_path)
        instrumental = return_dict["instrumentals"]
        vocals = return_dict["vocals"]
        input_audio = return_dict["input_audio"]

        if self.use_cache:
            os.makedirs(vocals_path,exist_ok=True)
            os.makedirs(instrumental_path,exist_ok=True)
            save_input_audio(vocals_file,vocals,to_int16=True)
            save_input_audio(instrumental_file,instrumental,to_int16=True)

        return vocals, instrumental, input_audio

def get_filename(*args,**kwargs):
    name = "_".join([str(arg) for arg in args]+[f"{k}={v}" for k,v in kwargs.items()])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache,cache_dir,num_threads,format) = arg
    
    model = Separator(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half="cuda" in str(device),
            use_cache=use_cache,
            cache_dir=cache_dir,
            num_threads = num_threads
            )
    vocals, instrumental, input_audio = model.run_inference(audio_path,format)
    del model
    gc_collect()

    return vocals, instrumental, input_audio
    
def split_audio(model_paths,audio_path,preprocess_models=[],postprocess_models=[],device="cuda",agg=10,use_cache=False,merge_type="mean",format="mp3",**kwargs):
    print(f"unused kwargs={kwargs}")
    merge_func = np.nanmedian if merge_type=="median" else np.nanmean
    num_threads = max(get_optimal_threads(-1),1)
    song_name = os.path.basename(audio_path).split(".")[0]
    cache_dir = CACHED_SONGS_DIR

    # preprocess input song to split reverb
    if len(preprocess_models):
        output_name = get_filename(*[os.path.basename(name).split(".")[0] for name in preprocess_models],agg=agg) + f".{format}"
        preprocessed_file = os.path.join(cache_dir,song_name,"preprocessing",output_name)
        
        # read from cache
        if os.path.isfile(preprocessed_file): input_audio = load_input_audio(preprocessed_file,mono=True)
        else: # preprocess audio
            for i,preprocess_model in enumerate(preprocess_models):
                output_name = get_filename(i,os.path.basename(preprocess_model).split(".")[0],agg=agg) + f".{format}"
                intermediary_file = os.path.join(cache_dir,song_name,"preprocessing",output_name)
                if os.path.isfile(intermediary_file):
                    if i==len(preprocess_model)-1: #last model
                        input_audio = load_input_audio(intermediary_file, mono=True)
                else:
                    args = (preprocess_model,audio_path,agg,device,False,CACHED_SONGS_DIR if i==0 else None,num_threads,format)
                    _, instrumental, input_audio = __run_inference_worker(args)
                    save_input_audio(intermediary_file,instrumental,to_int16=True)
                audio_path = intermediary_file

            save_input_audio(preprocessed_file,instrumental,to_int16=True)
        audio_path = preprocessed_file
        cache_dir = os.path.join(CACHED_SONGS_DIR,song_name)
    else:
        input_audio = load_input_audio(audio_path,mono=True)
        cache_dir = CACHED_SONGS_DIR
        
    # apply vocal separation
    wav_instrument = []
    wav_vocals = []

    for model_path in model_paths:
        args = (model_path,audio_path,agg,device,use_cache,cache_dir,num_threads,format)
        vocals, instrumental, _ = __run_inference_worker(args)
        wav_vocals.append(vocals[0])
        wav_instrument.append(instrumental[0])
    wav_instrument = merge_func(pad_audio(*wav_instrument),axis=0)
    wav_vocals = merge_func(pad_audio(*wav_vocals),axis=0)

    # postprocess vocals to reduce reverb
    if len(postprocess_models):
        vocals_name = get_filename("vocals",*[os.path.basename(name).split(".")[0] for name in model_paths],agg=agg) + f".{format}"
        vocals_file = os.path.join(cache_dir,"postprocessing",vocals_name)
        if not os.path.isfile(vocals_file): save_input_audio(vocals_file,(wav_vocals,vocals[-1]),to_int16=True)
        print("postprocessing...")        
        for i,postprocess_model in enumerate(postprocess_models):
            output_name = get_filename(i,os.path.basename(postprocess_model).split(".")[0],agg=agg) + f".{format}"
            intermediary_file = os.path.join(cache_dir,"postprocessing",output_name)
            if not os.path.isfile(intermediary_file):
                args = (postprocess_model,vocals_file,agg,device,False,None,num_threads,format)
                _, processed_audio, _ = __run_inference_worker(args)
                output_name = get_filename(i,os.path.basename(postprocess_model).split(".")[0],agg=agg) + f".{format}"
                save_input_audio(intermediary_file,processed_audio,to_int16=True)
                wav_vocals, _ = processed_audio
            vocals_file = intermediary_file

    instrumental = remix_audio((wav_instrument,instrumental[-1]),norm=True,to_int16=True,to_mono=True)
    vocals = remix_audio((wav_vocals,vocals[-1]),norm=True,to_int16=True,to_mono=True)

    return vocals, instrumental, input_audio

def main(): #uvr5_models,audio_path,device="cuda",agg=10,use_cache=False
    
    parser = argparse.ArgumentParser(description="processes audio to split vocal stems and reduce reverb/echo")
    
    parser.add_argument("uvr5_models", type=str, nargs="+", help="Path to models to use for processing")
    parser.add_argument(
        "-i", "--audio_path", type=str, help="path to audio file to process", required=True
    )
    parser.add_argument(
        "-p", "--preprocess_model", type=str, help="preprocessing model to improve audio", default=None
    )
    parser.add_argument(
        "-a", "--agg", type=int, default=10, help="aggressiveness score for processing (0-20)"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", choices=["cpu","cuda"], help="perform calculations on [cpu] or [cuda]"
    )
    parser.add_argument(
        "-m", "--merge_type", type=str, default="median", choices=["mean","median"], help="how to combine processed audio"
    )
    parser.add_argument(
        "-c", "--use_cache", type=bool, action="store_true", default=False, help="caches the results so next run is faster"
    )
    args = parser.parse_args()
    return split_audio(**vars(args))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()