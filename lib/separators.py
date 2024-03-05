import hashlib
import json
from multiprocessing.pool import ThreadPool
import os
from lib.utils import ObjectNamespace
import numpy as np
import torch
from tqdm import tqdm
from lib.mdx import MDXModel
from lib.uvr5_pack.constants import MDX_NET_FREQ_CUT
from lib.uvr5_pack.vr_network.model_param_init import ModelParameters
from lib.uvr5_pack.vr_network.nets_new import CascadedNet
from lib.uvr5_pack.vr_network.nets import CascadedASPPNet
from lib.audio import remix_audio
import librosa
from lib.uvr5_pack import spec_utils
class UVR5Base:
    
    def __init__(self, agg, model_path, device, is_half,**kwargs):
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
        mp = ModelParameters("lib/uvr5_pack/vr_network/modelparams/4band_v2.json")
        model = CascadedASPPNet(mp.param["bins"] * 2)
        cpk = torch.load(model_path, map_location=self.device)
        model.load_state_dict(cpk)
        model.eval()
        if is_half:
            model = model.half().to(device)
        else:
            model = model.to(device)

        self.mp = mp
        self.model = model
    
    @staticmethod
    def get_model_params(model_path):
        try:
            with open(model_path, 'rb') as f:
                f.seek(- 10000 * 1024, 2)
                model_hash = hashlib.md5(f.read()).hexdigest()
        except:
            model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

        print(model_hash)
        model_settings_json = os.path.splitext(model_path)[0]+".json"
        model_data_json = os.path.join(os.path.dirname(model_path),"model_data.json")

        if os.path.isfile(model_settings_json):
            return json.load(open(model_settings_json))
        elif os.path.isfile(model_data_json):
            with open(model_data_json,"r") as d:
                hash_mapper = json.loads(d.read())

            for hash, settings in hash_mapper.items():
                if model_hash in hash:
                    return settings
        return None

    def inference(self, X_spec, aggressiveness):
        """
        data ： dic configs
        """
        data = self.data
        device = self.device
        model = self.model

        def _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half=True
        ):
            model.eval()
            with torch.no_grad():
                preds = []

                for i in tqdm(range(n_window)):
                    start = i * roi_size
                    X_mag_window = X_mag_pad[
                        None, :, :, start : start + data["window_size"]
                    ]
                    X_mag_window = torch.from_numpy(X_mag_window)
                    if is_half:
                        X_mag_window = X_mag_window.half()
                    X_mag_window = X_mag_window.to(device)

                    pred = model.predict(X_mag_window, aggressiveness)

                    pred = pred.detach().cpu().numpy()
                    preds.append(pred[0])

                pred = np.concatenate(preds, axis=2)
            return pred

        def preprocess(X_spec):
            X_mag = np.abs(X_spec)
            X_phase = np.angle(X_spec)

            return X_mag, X_phase

        X_mag, X_phase = preprocess(X_spec)

        coef = X_mag.max()
        X_mag_pre = X_mag / coef

        n_frame = X_mag_pre.shape[2]
        pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, data["window_size"], model.offset)
        n_window = int(np.ceil(n_frame / roi_size))

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        if list(model.state_dict().values())[0].dtype == torch.float16:
            is_half = True
        else:
            is_half = False
        pred = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred = pred[:, :, :n_frame]

        if data["tta"]:
            pad_l += roi_size // 2
            pad_r += roi_size // 2
            n_window += 1

            X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

            pred_tta = _execute(
                X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
            )
            pred_tta = pred_tta[:, :, roi_size // 2 :]
            pred_tta = pred_tta[:, :, :n_frame]

            return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
        else:
            return pred * coef, X_mag, np.exp(1.0j * X_phase)

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
        print(f"vocals done: {wav_vocals.shape}")
        return_dict["vocals"] = remix_audio((wav_vocals,return_dict["sr"]),to_int16=True,axis=0)
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
        print(f"instruments done: {wav_instrument.shape}")
        return_dict["instrumentals"] = remix_audio((wav_instrument,return_dict["sr"]),to_int16=True,axis=0)
        return return_dict["instrumentals"] 
    
    def process_audio(self,y_spec_m,v_spec_m,input_high_end,input_high_end_h):
        return_dict = {
            "sr": self.mp.param["sr"]
        }
        
        with ThreadPool(2) as pool:
            pool.apply(self.process_vocals, args=(v_spec_m,input_high_end,input_high_end_h,return_dict))
            pool.apply(self.process_instrumental, args=(y_spec_m,input_high_end,input_high_end_h,return_dict))
 
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
            pred, X_mag, X_phase = self.inference(X_spec_m, aggressiveness)

    #     return pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h

    # def run_process_audio(self, pred, X_mag, X_phase, X_spec_m, input_high_end,input_high_end_h):
        # Postprocess
        if self.data["postprocess"]:
            pred_inv = np.clip(X_mag - pred, 0, np.inf)
            pred = spec_utils.mask_silence(pred, pred_inv)
        y_spec_m = pred * X_phase
        v_spec_m = X_spec_m - y_spec_m

        
        return_dict = self.process_audio(y_spec_m,v_spec_m,input_high_end,input_high_end_h)
        return_dict["input_audio"] = input_audio
        
        return return_dict

class UVR5New(UVR5Base):
    def __init__(self, agg, model_path, device, is_half, dereverb, **kwargs):
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
        mp = ModelParameters("lib/uvr5_pack/vr_network/modelparams/4band_v3.json")
        nout = 64 if dereverb else 48
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
    
class MDXNet:
    def __init__(self, model_path, chunks=15,denoise=False,num_threads=1,device="cpu",**kwargs):

        self.chunks = chunks
        self.sr = 44100
        
        self.args = ObjectNamespace(**kwargs)
        self.denoise = denoise
        self.num_threads = num_threads

        self.device = device

        self.is_mdx_ckpt = "ckpt" in model_path

        self.model = MDXModel(model_path, device=self.device,chunks=self.chunks,margin=self.sr)

    def __del__(self):
        del self.model

    def process_audio(self,primary,secondary,target_sr=None):
        target_sr =  self.sr if target_sr is None else target_sr
        # foreground is processed data
        vocals,instrumental = (secondary,primary) if "instrument" in self.model.params.stem_name.lower() else (primary,secondary)
        with ThreadPool(2) as pool:
            results = pool.starmap(remix_audio, [
                ((instrumental,self.sr),target_sr,False,True,self.sr!=target_sr,True),
                ((vocals,self.sr),target_sr,False,True,self.sr!=target_sr,True)
            ])

        return_dict = {
            "sr": target_sr,
            "instrumentals": results[0],
            "vocals": results[1]
        }
        return return_dict
    
    def run_inference(self, audio_path):
        
        mdx_net_cut = True if self.model.params.stem_name in MDX_NET_FREQ_CUT else False
        mix, raw_mix, samplerate = prepare_mix(audio_path, self.model.chunks, self.model.margin, mdx_net_cut=mdx_net_cut)
        wave_processed = self.model.demix_base(mix, is_ckpt=self.is_mdx_ckpt)[0]
        
    
        raw_mix = self.model.demix_base(raw_mix, is_match_mix=True)[0] if mdx_net_cut else raw_mix

        return_dict = self.process_audio(primary=wave_processed,secondary=(raw_mix-wave_processed),target_sr=samplerate)
        return_dict["input_audio"] = (raw_mix, samplerate)

        return return_dict

    
def prepare_mix(mix, chunk_set, margin_set, mdx_net_cut=False):

    samplerate = 44100

    if not isinstance(mix, np.ndarray):
        mix, samplerate = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    def get_segmented_mix(chunk_set=chunk_set):
        segmented_mix = {}
        
        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set*44100
        assert not margin == 0, 'margin cannot be zero!'
        
        if margin > chunk_size:
            margin = chunk_size
        if chunk_set == 0 or samples < chunk_size:
            chunk_size = samples
        
        counter = -1
        for skip in range(0, samples, chunk_size):
            counter+=1
            s_margin = 0 if counter == 0 else margin
            end = min(skip+chunk_size+margin, samples)
            start = skip-s_margin
            segmented_mix[skip] = mix[:,start:end].copy()
            if end == samples:
                break
            
        return segmented_mix

    
    segmented_mix = get_segmented_mix()
    raw_mix = get_segmented_mix(chunk_set=0) if mdx_net_cut else mix
    return segmented_mix, raw_mix, samplerate