import numpy as np, torch, sys, os
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
import os, traceback, faiss, librosa
from scipy import signal

# from tqdm import tqdm

from pitch_extraction import FeatureExtractor

from webui.audio import load_input_audio, remix_audio
from webui import config, get_cwd

CWD = get_cwd()

from webui.utils import gc_collect, get_filenames

# torchcrepe = lazyload("torchcrepe")  # Fork Feature. Crepe algo for training and preprocess
# torch = lazyload("torch")
# rmvpe = lazyload("rmvpe")

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class VC(FeatureExtractor):

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        t0 = ttime()
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = feats.clone()
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch != None and pitchf != None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def process_t(self, t, s, window, audio_pad, pitch, pitchf, times, index, big_npy, index_rate, version, protect, t_pad_tgt, if_f0, sid, model, net_g):
        t = t // window * window
        if if_f0 == 1:
            return self.vc(
                model,
                net_g,
                sid,
                audio_pad[s : t + t_pad_tgt + window],
                pitch[:, s // window : (t + t_pad_tgt) // window],
                pitchf[:, s // window : (t + t_pad_tgt) // window],
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[t_pad_tgt : -t_pad_tgt]
        else:
            return self.vc(
                model,
                net_g,
                sid,
                audio_pad[s : t + t_pad_tgt + window],
                None,
                None,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[t_pad_tgt : -t_pad_tgt]

    def pipeline(self, model, net_g, sid, audio, times, f0_up_key, f0_method, merge_type,
            file_index, index_rate, if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate,
            version, protect, crepe_hop_length, f0_autotune, rmvpe_onnx, f0_file=None, f0_min=50, f0_max=1100):
        
        try:
            if file_index == "":
                print("File index was empty.")
                index = None
                big_npy = None
            else:
                if os.path.exists(file_index):
                    sys.stdout.write(f"Attempting to load {file_index}....\n")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"Attempting to load {file_index}.... (despite it not existing)\n")
                    sys.stdout.flush()
                index = faiss.read_index(file_index)
                sys.stdout.write(f"loaded index: {index}\n")
                big_npy = index.reconstruct_n(0, index.ntotal)
        except Exception as e:
            print(f"Could not open Faiss index file for reading. {e}")
            index = None
            big_npy = None

        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            
            for t in range(self.t_center, audio.shape[0], self.t_center):
                abs_audio_sum = np.abs(audio_sum[t - self.t_query : t + self.t_query])
                min_abs_audio_sum = abs_audio_sum.min()
                opt_ts.append(t - self.t_query + np.where(abs_audio_sum == min_abs_audio_sum)[0][0])

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None

        if f0_file is not None:
            try:
                with open(f0_file.name, "r") as f:
                    inp_f0 = np.array([list(map(float, line.split(","))) for line in f.read().strip("\n").split("\n")], dtype="float32")
            except:
                traceback.print_exc()

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None

        if if_f0:
            pitch, pitchf = self.get_f0(
                audio_pad, p_len, f0_up_key, f0_method, merge_type,
                filter_radius, crepe_hop_length, f0_autotune, rmvpe_onnx, inp_f0, f0_min, f0_max)
            
            pitch = pitch[:p_len].astype(np.int64 if self.device != 'mps' else np.float32)
            pitchf = pitchf[:p_len].astype(np.float32)
            pitch = torch.from_numpy(pitch).to(self.device).unsqueeze(0)
            pitchf = torch.from_numpy(pitchf).to(self.device).unsqueeze(0)

        t2 = ttime()
        times[1] += t2 - t1

        # with tqdm(total=len(opt_ts), desc="Processing", unit="window") as pbar:
        for i, t in enumerate(opt_ts):
            t = t // self.window * self.window
            start = s
            end = t + self.t_pad2 + self.window
            audio_slice = audio_pad[start:end]
            pitch_slice = pitch[:, start // self.window:end // self.window] if if_f0 else None
            pitchf_slice = pitchf[:, start // self.window:end // self.window] if if_f0 else None
            audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            s = t
                # pbar.update(1)
                # pbar.refresh()

        audio_slice = audio_pad[t:]
        pitch_slice = pitch[:, t // self.window:] if if_f0 and t is not None else pitch
        pitchf_slice = pitchf[:, t // self.window:] if if_f0 and t is not None else pitchf
        audio_opt.append(self.vc(model, net_g, sid, audio_slice, pitch_slice, pitchf_slice, times, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        
        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)

        max_int16 = 32768
        audio_max = max(np.abs(audio_opt).max() / 0.99, 1)
        audio_opt = (audio_opt * max_int16 / audio_max).astype(np.int16)

        gc_collect()

        print("Returning completed audio...")
        print("-------------------")
        
        return audio_opt

def get_vc(model_path,config,device=None):
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    
    if version == "v1":
        if if_f0 == 1:
            from lib.infer_pack.models import SynthesizerTrnMs256NSFsid
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            from lib.infer_pack.models import SynthesizerTrnMs256NSFsid_nono
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            from lib.infer_pack.models import SynthesizerTrnMs768NSFsid
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            from lib.infer_pack.models import SynthesizerTrnMs768NSFsid_nono
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(device if device else config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    hubert_model = load_hubert(config)
    model_name = os.path.basename(model_path).split(".")[0]
    index_file = get_filenames(root=os.path.join(CWD,"models","RVC"),folder=".index",exts=["index"],name_filters=[model_name])
    return {"vc": vc, "cpt": cpt, "net_g": net_g, "hubert_model": hubert_model,"model_name": model_name,
            "file_index": index_file[0] if len(index_file) else ""}

def load_hubert(config):
    try:
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(CWD,"models","hubert_base.pt")],
            suffix="",
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(config.device)
        if config.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        hubert_model.eval()
        return hubert_model
    except Exception as e:
        print(e)
        return None

def vc_single(
        cpt=None,
        net_g=None,
        vc=None,
        hubert_model=None,
    sid=0,
    input_audio=None,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="crepe",
    merge_type="median",
    file_index="",  # .index file
    index_rate=.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=.25,
    protect=0.33,
    crepe_hop_length=160,
    f0_autotune=False,
    is_onnx=False,
    config=config,
    **kwargs #prevents function from breaking
):
    print(f"vc_single unused args: {kwargs}")
    if hubert_model == None:
        hubert_model = load_hubert(config)

    if not (cpt and net_g and vc and hubert_model):
        return None

    tgt_sr = cpt["config"][-1]
    
    version = cpt.get("version", "v1")

    if input_audio is None and input_audio_path is None:
        return None
    f0_up_key = int(f0_up_key)
    try:
        audio = input_audio[0] if input_audio is not None else load_input_audio(input_audio_path, 16000)
        
        audio,_ = remix_audio((audio,input_audio[1] if input_audio is not None else 16000), target_sr=16000, norm=True,  to_mono=True)

        times = [0, 0, 0]
        if_f0 = cpt.get("f0", 1)
        
        """
        model, net_g, sid, audio, times, f0_up_key, f0_method,
            file_index, index_rate, if_f0, filter_radius, tgt_sr, resample_sr, rms_mix_rate,
            version, protect, crepe_hop_length, f0_autotune, rmvpe_onnx
        """
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method if len(f0_method)>1 else f0_method[0], # more than 1 f0_method in list means hybrid
            merge_type,
            file_index,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length, f0_autotune, is_onnx,
            f0_file=f0_file,
        )
        
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        print(index_info)
        
        return (audio_opt, resample_sr if resample_sr >= 16000 and tgt_sr != resample_sr else tgt_sr)
    except Exception as info:
        print(info)
        return None