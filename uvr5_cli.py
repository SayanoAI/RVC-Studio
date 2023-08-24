import argparse
import multiprocessing
from multiprocessing.pool import ThreadPool
import os, sys, torch, warnings
from types import SimpleNamespace

from tqdm import tqdm

from webui_utils import gc_collect, load_input_audio, remix_audio, save_input_audio

now_dir = os.getcwd()
sys.path.append(now_dir)

warnings.filterwarnings("ignore")
import librosa
import numpy as np
from lib.uvr5_pack.lib_v5 import spec_utils
from lib.uvr5_pack.utils import inference
from lib.uvr5_pack.lib_v5.model_param_init import ModelParameters
import soundfile as sf
from lib.uvr5_pack.lib_v5.nets_new import CascadedNet
from lib.uvr5_pack.lib_v5 import nets_61968KB as nets


class UVR5Base:
    def __init__(self, agg, model_path, device, is_half):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("lib/uvr5_pack/lib_v5/modelparams/4band_v2.json")
        model = nets.CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    
    def process_vocals(self,v_spec_m,input_high_end,input_high_end_h,return_dict={}):
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], v_spec_m, input_high_end, self.mp
            )
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(
                v_spec_m, self.mp, input_high_end_h, input_high_end_
            )
        else:
            wav_vocals = spec_utils.cmb_spectrogram_to_wave(v_spec_m, self.mp)
        print("vocals done!")
        return_dict["vocals"] = remix_audio((wav_vocals,return_dict["sr"]),norm=True,to_int16=True,to_mono=True)
        return return_dict["vocals"]
    
    def process_instrumental(self,y_spec_m,input_high_end,input_high_end_h,return_dict={}):
        if self.data["high_end_process"].startswith("mirroring"):
            input_high_end_ = spec_utils.mirroring(
                self.data["high_end_process"], y_spec_m, input_high_end, self.mp
            )
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(
                y_spec_m, self.mp, input_high_end_h, input_high_end_
            )
        else:
            wav_instrument = spec_utils.cmb_spectrogram_to_wave(y_spec_m, self.mp)
        print("instruments done!")
        return_dict["instrumentals"] = remix_audio((wav_instrument,return_dict["sr"]),norm=True,to_int16=True,to_mono=True)
        return return_dict["instrumentals"] 
    
    def process_audio(self,y_spec_m,v_spec_m,input_high_end,input_high_end_h):
        return_dict = {
            "sr": self.mp.param["sr"]
        }
        
        with ThreadPool(2) as pool:
            pool.apply(self.process_vocals, args=(v_spec_m,input_high_end,input_high_end_h,return_dict))
            pool.apply(self.process_instrumental, args=(y_spec_m,input_high_end,input_high_end_h,return_dict))
 
        # thread1 = threading.Thread(target=self.process_vocals, args=(v_spec_m,input_high_end,input_high_end_h,return_dict))
        # thread2 = threading.Thread(target=self.process_instrumental, args=(y_spec_m,input_high_end,input_high_end_h,return_dict))
        # thread1.start(),thread2.start()
        # thread1.join(),thread2.join()
        return return_dict

    def run_inference(self, music_file):
        X_wave,  X_spec_s = {}, {}
        bands_n = len(self.mp.param["band"])
        # print(bands_n)
        for d in range(bands_n, 0, -1):
            bp = self.mp.param["band"][d]
            if d == bands_n:  # high-end band
                input_audio = librosa.core.load(music_file,sr=bp["sr"],res_type=bp["res_type"])
                X_wave[d] = input_audio[0]
                if X_wave[d].ndim == 1:
                    X_wave[d] = np.asfortranarray([X_wave[d], X_wave[d]])
            else:  # lower bands
                X_wave[d] = librosa.core.resample(
                    X_wave[d + 1],
                    self.mp.param["band"][d + 1]["sr"],
                    bp["sr"],
                    res_type=bp["res_type"],
                )
            # Stft of wave source
            X_spec_s[d] = spec_utils.wave_to_spectrogram_mt(
                X_wave[d],
                bp["hl"],
                bp["n_fft"],
                self.mp.param["mid_side"],
                self.mp.param["mid_side_b2"],
                self.mp.param["reverse"],
            )
            # pdb.set_trace()
            if d == bands_n and self.data["high_end_process"] != "none":
                input_high_end_h = (bp["n_fft"] // 2 - bp["crop_stop"]) + (
                    self.mp.param["pre_filter_stop"] - self.mp.param["pre_filter_start"]
                )
                input_high_end = X_spec_s[d][
                    :, bp["n_fft"] // 2 - input_high_end_h : bp["n_fft"] // 2, :
                ]

        X_spec_m = spec_utils.combine_spectrograms(X_spec_s, self.mp)
        aggresive_set = float(self.data["agg"] / 100)
        aggressiveness = {
            "value": aggresive_set,
            "split_bin": self.mp.param["band"][1]["crop_stop"],
        }

        # pred, X_mag, X_phase = run_inference(X_spec_m, self.device, self.models, aggressiveness, self.data)
        with torch.no_grad():
            pred, X_mag, X_phase = inference(
                X_spec_m, self.device, self.model, aggressiveness, self.data
            )

    #     return pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h

    # def run_process_audio(self, pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h):
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        
        return_dict = self.process_audio(y_spec_m,v_spec_m,input_high_end,input_high_end_h)
        
        return return_dict

class UVR5Dereverb(UVR5Base):
    def __init__(self, agg, model_path, device, is_half):
        self.model_path = model_path
        self.device = device
        self.data = {
            # Processing Options
            "postprocess": False,
            "tta": False,
            # Constants
            "window_size": 512,
            "agg": agg,
            "high_end_process": "mirroring",
        }
        mp = ModelParameters("lib/uvr5_pack/lib_v5/modelparams/4band_v3.json")
        nout = 64 if "DeReverb" in model_path else 48
        model = CascadedNet(mp.param["bins"] * 2, nout)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    

class Conv_TDF_net_trim:
    def __init__(
        self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(Conv_TDF_net_trim, self).__init__()

        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(
            device
        )
        self.target_name = target_name
        self.blender = "blender" in model_name
        self.dim_c = 4

        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        ).to(device)

        self.n = L // 2
        self.device=device

    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.window,
            center=True,
            return_complex=True,
        )
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        return x[:, :, : self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = (
            self.freq_pad.repeat([x.shape[0], 1, 1, 1])
            if freq_pad is None
            else freq_pad
        )
        x = torch.cat([x, freq_pad], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x = x.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x = x.permute([0, 2, 3, 1])
        x = x.contiguous()
        x = torch.view_as_complex(x)
        x = torch.istft(
            x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True
        )
        return x.reshape([-1, c, self.chunk_size])

class MDXNetDereverb:
    def __init__(self, model_path, chunks=15, **kwargs):
        self.onnx = model_path
        self.shifts = 10  #'Predict with randomised equivariant stabilisation'
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        self.chunks = chunks
        # self.margin = 44100
        self.margin = 16000
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.args = SimpleNamespace(**kwargs)
        self.device = "cpu" #self.args.device #cuda doesn't work for some reason
        self.model_ = Conv_TDF_net_trim(
            device=self.device,
            model_name="Conv-TDF",
            target_name="vocals",
            L=11,
            dim_f=self.dim_f,
            dim_t=self.dim_t,
            n_fft=self.n_fft,
        )
        self.mp = SimpleNamespace(param={"sr": self.margin})
       
        import onnxruntime as ort

        print(ort.get_available_providers())
        self.model = ort.InferenceSession(
            self.onnx,
            # os.path.join(args.onnx, self.model_.target_name + ".onnx"),
            providers=[
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        print(f"onnx load done: {dir(self.model.get_modelmeta())} ")

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.margin
        assert not margin == 0, "margin cannot be zero!"
        chunk_size = self.chunks * margin
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []

        for mix in tqdm(mixes,"Processing"):
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred).to(self.device))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0]).to(self.device)
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .numpy()[:, :-pad]
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                sources.append(tar_signal[:, start:end])

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        return _sources
    
    def process_audio(self,instrumental,vocals):
        sr = self.mp.param["sr"]
        return_dict = {"sr": sr}
        
        with ThreadPool(2) as pool:
            results = pool.starmap(remix_audio, [
                ((instrumental,sr),sr,True,True,False,True),
                ((vocals,sr),sr,True,True,False,True)
            ])

        return_dict["instrumental"] = results[0]
        return_dict["vocals"] = results[1]
        return return_dict
    
    def run_inference(self, audio_path):
        mix, _ = librosa.load(audio_path, mono=False, sr=self.margin)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T

        return_dict = self.process_audio(instrumental=mix-opt,vocals=opt)
        
        return return_dict

class UVR5_Model:
    def __init__(self, model_path, use_cache=False, **kwargs):

        # self.models = {}
        # for model_path in uvr5_models:
        if "MDX" in model_path:
            self.model = MDXNetDereverb(model_path=model_path,**kwargs)
        elif "VR_Models":
            self.model = UVR5Dereverb(model_path=model_path,**kwargs) if any([
                ele in model_path.lower() for ele in ["echo","noise","reverb"]
                ]) else UVR5Base(model_path=model_path,**kwargs)
            # self.models[model_path] = __instance__
        self.use_cache = use_cache
        self.model_path = model_path
        self.args = kwargs
    
    # cleanup memory
    def __del__(self):
        gc_collect(self.model)

    def run_inference(self, audio_path):
        name = get_filename(self.model_path,audio_path,**self.args)
        
        # handles loading of previous processed data
        music_dir = os.path.join(os.path.dirname(audio_path))
        vocals_path = os.path.join(music_dir,".cache",".vocals")
        instrumental_path = os.path.join(music_dir,".cache",".instrumental")
        vocals_file = os.path.join(vocals_path,name)
        instrumental_file = os.path.join(instrumental_path,name)
        os.makedirs(vocals_path,exist_ok=True)
        os.makedirs(instrumental_path,exist_ok=True)
        input_audio = load_input_audio(audio_path,mono=True,sr=self.model.mp.param["sr"])

        if os.path.exists(instrumental_file) and os.path.exists(vocals_file):
            vocals = load_input_audio(vocals_file,mono=True)
            instrumental = load_input_audio(instrumental_file,mono=True)
            # input_audio = load_input_audio(audio_path,mono=True)
            return vocals, instrumental, input_audio
        
        return_dict = self.model.run_inference(audio_path)
        instrumental = return_dict["instrumentals"]
        vocals = return_dict["vocals"]
        # instrumental = remix_audio((return_dict["instrumentals"],return_dict["sr"]),norm=True,to_int16=True,to_mono=True)
        # vocals = remix_audio((return_dict["vocals"],return_dict["sr"]),norm=True,to_int16=True,to_mono=True)

        if self.use_cache:
            save_input_audio(vocals_file,vocals,to_int16=True)
            save_input_audio(instrumental_file,instrumental,to_int16=True)

        return vocals, instrumental, input_audio

def get_filename(model_path,audio_path,**args):
    name = "_".join([
        str(v) for v in args.values() if not hasattr(v, '__dict__')]+[
            os.path.basename(model_path).split(".")[0],os.path.basename(audio_path)
            ])
    return name

def __run_inference_worker(arg):
    (model_path,audio_path,agg,device,use_cache) = arg
    
    model = UVR5_Model(
            agg=agg,
            model_path=model_path,
            device=device,
            is_half=device=="cuda",
            use_cache=use_cache
            )
    vocals, instrumental, input_audio = model.run_inference(audio_path)

    return vocals, instrumental, input_audio
    
def split_audio(uvr5_models,audio_path,preprocess_model=None,device="cuda",agg=10,use_cache=False):
    
    pooled_data = []

    if preprocess_model:
        model = UVR5_Model(
            agg=agg,
            model_path=preprocess_model,
            device=device,
            is_half=device=="cuda",
            use_cache=use_cache
            )
        print("preprocessing")
        _, instrumental, input_audio = model.run_inference(audio_path)
        print(f"{instrumental[0].max()}, {instrumental[0].min()}, sr={instrumental[1]}")
        # saves preprocessed file to cache and use as input
        name = get_filename(preprocess_model,"",device=device,agg=agg)
        music_dir = os.path.join(os.path.dirname(audio_path))
        processed_path = os.path.join(music_dir,".cache",name)
        os.makedirs(processed_path,exist_ok=True)
        audio_path = os.sep.join([processed_path,os.path.basename(audio_path)])
        if not os.path.exists(audio_path):
            save_input_audio(audio_path,instrumental,to_int16=True)
    else:
        input_audio = load_input_audio(audio_path,mono=True)
    
    args = [(model_path,audio_path,agg,device,use_cache) for model_path in uvr5_models]
    with multiprocessing.Pool(len(args)) as pool:
        pooled_data = pool.map(__run_inference_worker,args)
    # pool.join()
        
    wav_instrument = []
    wav_vocals = []

    for ( vocals, instrumental, _) in pooled_data:
        wav_vocals.append(vocals[0])
        wav_instrument.append(instrumental[0])

    instrumental = remix_audio((np.mean(wav_instrument,axis=0),input_audio[1]),norm=True,to_int16=True,to_mono=True)
    vocals = remix_audio((np.mean(wav_vocals,axis=0),input_audio[1]),norm=True,to_int16=True,to_mono=True)

    return vocals, instrumental, input_audio

def main(): #uvr5_models,audio_path,device="cuda",agg=10,use_cache=False
    
    parser = argparse.ArgumentParser(description="processes audio to split vocal stems and reduce reverb/echo")
    
    parser.add_argument("uvr5_models", type=str, nargs="+", help="Path to models to use for processing", required=True)
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
        "-c", "--use_cache", type=bool, action="store_true", default=False, help="caches the results so next run is faster"
    )
    args = parser.parse_args()
    return split_audio(**vars(args))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()