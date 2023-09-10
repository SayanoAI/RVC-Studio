import argparse
import os, sys, torch, warnings

from lib.separators import MDXNet, UVR5Base, UVR5New
from webui.audio import load_input_audio, remix_audio, save_input_audio
from webui.utils import gc_collect

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.append(CWD)
CACHE_DIR = os.path.join(CWD,".cache","songs")

warnings.filterwarnings("ignore")
import librosa
import numpy as np

class Separator:
    def __init__(self, model_path, use_cache=False, device="cpu", cache_dir=CACHE_DIR, **kwargs):
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

    def run_inference(self, audio_path):
        song_name = get_filename(
            os.path.basename(self.model_path).split(".")[0],
            **self.args) + ".mp3"
        
        # handles loading of previous processed data
        music_dir = os.path.join(self.cache_dir,os.path.basename(audio_path).split(".")[0])
        vocals_path = os.path.join(music_dir,".vocals")
        instrumental_path = os.path.join(music_dir,".instrumental")
        vocals_file = os.path.join(vocals_path,song_name)
        instrumental_file = os.path.join(instrumental_path,song_name)
        os.makedirs(vocals_path,exist_ok=True)
        os.makedirs(instrumental_path,exist_ok=True)
        # input_audio = load_input_audio(audio_path,mono=True)

        if os.path.exists(instrumental_file) and os.path.exists(vocals_file):
            vocals = load_input_audio(vocals_file,mono=True)
            instrumental = load_input_audio(instrumental_file,mono=True)
            input_audio = load_input_audio(audio_path,mono=True)
            return vocals, instrumental, input_audio
        
        return_dict = self.model.run_inference(audio_path)
        instrumental = return_dict["instrumentals"]
        vocals = return_dict["vocals"]
        input_audio = return_dict["input_audio"]

        if self.use_cache:
            save_input_audio(vocals_file,vocals,to_int16=True)
            save_input_audio(instrumental_file,instrumental,to_int16=True)

        return vocals, instrumental, input_audio

def get_filename(*args,**kwargs):
    name = "_".join([str(arg) for arg in args]+[f"{k}={v}" for k,v in kwargs.items()])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache,cache_dir) = arg
    
    model = Separator(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=device=="cuda",
            use_cache=use_cache,
            cache_dir=cache_dir
            )
    vocals, instrumental, input_audio = model.run_inference(audio_path)
    del model
    gc_collect()

    return vocals, instrumental, input_audio
    
def split_audio(uvr5_models,audio_path,preprocess_model=None,device="cuda",agg=10,use_cache=False,merge_type="mean"):
    cache_dir = CACHE_DIR #default cache dir
    if preprocess_model:
        song_name = os.path.basename(audio_path).split(".")[0]
        output_name = get_filename(os.path.basename(preprocess_model).split(".")[0],agg=agg) + ".mp3"
        preprocess_path = os.path.join(CACHE_DIR,song_name,output_name)
        if os.path.exists(preprocess_path): input_audio = load_input_audio(preprocess_path,mono=True)
        else:
            model = Separator(
                agg=agg,
                model_path=preprocess_model,
                device=device,
                is_half="cuda" in device,
                use_cache=use_cache,
                num_threads = max(os.cpu_count()//2,1)
                )
            print("preprocessing")
            _, instrumental, input_audio = model.run_inference(audio_path)
            # saves preprocessed file to cache and use as input
            save_input_audio(preprocess_path,instrumental,to_int16=True)
            del model
            gc_collect()
        cache_dir = os.path.join(CACHE_DIR,song_name)
        audio_path = preprocess_path
    else:
        input_audio = load_input_audio(audio_path,mono=True)
        
    wav_instrument = []
    wav_vocals = []
    max_len = 0

    for model_path in uvr5_models:
        args = (model_path,audio_path,agg,device,use_cache,cache_dir)
        vocals, instrumental, _ = __run_inference_worker(args)
        wav_vocals.append(vocals[0])
        wav_instrument.append(instrumental[0])
        max_len = max(max_len,len(vocals[0]),len(instrumental[0]))

    merge_func = np.nanmedian if merge_type=="median" else np.nanmean
    wav_instrument = merge_func([librosa.util.pad_center(wav,max_len) for wav in wav_instrument],axis=0)
    wav_vocals = merge_func([librosa.util.pad_center(wav,max_len) for wav in wav_vocals],axis=0)
    instrumental = remix_audio((wav_instrument,instrumental[1]),norm=True,to_int16=True,to_mono=True)
    vocals = remix_audio((wav_vocals,vocals[1]),norm=True,to_int16=True,to_mono=True)

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