import os
import logging
from lib import BASE_MODELS_DIR, config
from lib.model_utils import change_rms, load_hubert

from pitch_extraction import FeatureExtractor
from lib.utils import gc_collect, get_filenames

logger = logging.getLogger(__name__)

import fairseq
import numpy as np
import torch
import torch.nn.functional as F

if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml


# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试
class RVC(FeatureExtractor):
    def __init__(self, model_path, config, onnx=False, device=None):

        cpt = torch.load(model_path, map_location="cpu")
        tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        version = cpt.get("version", "v1")
        self.sid = 0
        
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
        hubert_model = load_hubert(config)
        model_name = os.path.basename(model_path).split(".")[0]
        index_files = get_filenames(root=os.path.join(BASE_MODELS_DIR,"RVC"),folder=".index",exts=["index"],name_filters=[model_name])
        file_index = index_files.pop() if len(index_files) else ""

        self.cpt = cpt
        self.net_g = net_g
        self.hubert_model = hubert_model
        self.model_name = model_name
        self.index, self.big_npy = self.load_index(file_index)
        self.tgt_sr = tgt_sr
        self.if_f0 = if_f0
        self.version = version
        super().__init__(tgt_sr, config, onnx) # initiate Feature Extraction

    def __del__(self):
        super().__del__()
        del self.cpt, self.net_g, self.hubert_model, self.index, self.big_npy
        gc_collect()

    # def process_input(self, x: np.ndarray, **kwargs) -> np.ndarray:
    def vc(self, x: np.ndarray, **kwargs) -> np.ndarray:
        index_rate = kwargs.pop("index_rate",.5)
        protect = kwargs.pop("protect",.5)
        rms_mix_rate = kwargs.pop("rms_mix_rate",1.)
    
        feats = torch.from_numpy(x.copy())
        feats = feats.view(1, -1)
        if config.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        feats = feats.to(self.device)

        with torch.no_grad():
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if self.version == "v1" else 12,
            }
            logits = self.hubert_model.extract_features(**inputs)
            feats = (
                self.hubert_model.final_proj(logits[0]) if self.version == "v1" else logits[0]
            )

        if protect < 0.5 and self.if_f0:
            feats0 = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if self.index is not None and self.big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float16")

            score, ix = self.index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        if self.if_f0 == 1:
            pitch, pitchf = self.get_f0(x, **kwargs)
            p_len = min(feats.shape[1], pitch.shape[0])
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            pitch = torch.from_numpy(pitch).to(self.device).unsqueeze(0)
            pitchf = torch.from_numpy(pitchf).to(self.device).unsqueeze(0)
            if protect < 0.5:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)
                feats = feats * pitchff + feats0 * (1 - pitchff)
                del pitchff
        else:
            pitch, pitchf = None, None
            p_len = feats.shape[1]
       
        p_len = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([self.sid]).to(self.device)
        with torch.no_grad():
            if self.is_half: feats = feats.to(torch.half)
            if self.if_f0 == 1:
                # print("process_output",feats,p_len,pitch,pitchf)
                # print(12222222222,feats.dtype,pitch.dtype,pitchf.dtype,sid.dtype,self.is_half)
                infered_audio = (
                    self.net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0].data
                )
            else:
                infered_audio = (
                    self.net_g.infer(feats, p_len, sid)[0][0, 0].data
                )

            audio_opt = infered_audio.cpu().float().numpy()
            if rms_mix_rate < 1.:
                audio_opt = change_rms(x, self.sr, audio_opt, self.tgt_sr, rms_mix_rate)

            del feats, p_len, sid, pitch, pitchf, infered_audio
        return audio_opt