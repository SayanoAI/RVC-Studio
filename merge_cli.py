#!/usr/bin/python

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from safetensors import safe_open, torch as st_torch

parser = argparse.ArgumentParser(description="Merge models with weighted similarity")
parser.add_argument("models", type=str, nargs="+", help="Path to models")
parser.add_argument("--out", type=str, help="Output file name, without extension", default=None, required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--extension", type=str, help="Output file extension", default="safetensors", required=False)
parser.add_argument("--skip", type=str, help="which portion to skip: vae|clip|unet", default="", required=False)
parser.add_argument("--small", action="store_true", help="compress output to float16", default=True, required=False)
parser.add_argument("--method", type=str, help="which method to use: sum|cosine|max|scale", required=False)
parser.add_argument("--s", type=str, help="comma separated scaling for each model, single value to use the same for all models, keep blank for dynamic scaling", required=False)
parser.add_argument("--weight", type=float, help="weight multiplier for end model", default=1.0, required=False)
parser.add_argument("--verbose", action="store_true", help="whether to output logs", default=False, required=False)
parser.add_argument("--prune", action="store_true", help="whether to prune the model of extraneous keys (i.e. vae, training gradients)", default=False, required=False)
args = parser.parse_args()
       
def get_alpha(A,B,s=0.5,verbose=False):
    try:
        A = A if A.ndim==2 else A.flatten().unsqueeze(0)
        B = B if B.ndim==2 else B.flatten().unsqueeze(0)
        TA = torch.trace(A@A.T)
        TB = torch.trace(B@B.T)
        TAB = torch.trace(A@B.T)
        a=TB-2*TAB+TA
        b=2*(TAB-TA)
        c=s*(TA-TB)
        alpha=(-b - torch.sqrt(b**2-4*a*c))/(2*a) if TA>TB else (-b + torch.sqrt(b**2-4*a*c))/(2*a) #use quadratic formula to find alpha
        TC = TA + s*(TB-TA)
        if verbose:
            print(f"alpha={alpha} s={s} TC={TC}")
        return torch.clamp(torch.nan_to_num(alpha, nan=s),min=0,max=1).item()
    except Exception as e:
        print(e)
        return s
   
def get_scales(N):
    assert N>0, "number of models have to be greater than 0"
    s = np.ones(N-1)
    for i in range(N-1):
        s[-(1+i)] /= float(N-i)
    print(f"scales={s}")
    return s
        
def safetensors_load(ckpt, map_location="cpu"):
    sd = {}
    name, extension = os.path.splitext(ckpt)
    if extension.lower() == ".safetensors":
        with safe_open(ckpt, framework="pt", device=map_location) as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return {'state_dict': sd},os.path.basename(name)
    else:
        return torch.load(ckpt, map_location=torch.device(map_location)),os.path.basename(name)
    
def loadModelWeights(mPath):
    model,name = safetensors_load(mPath, map_location=args.device)
    
    try:
        theta = model["weight"]
        configs = [(key,model[key]) for key in model if key!="weight"]
        print(configs)
    except: theta = configs = model
    return theta,name,configs

def is_clip_key(key):
    return "text_model" in key

def is_vae_key(key):
    return key.startswith("first_stage_model")

def is_unet_key(key):
    return "diffusion_model" in key

def prune_model(model):
    for key in list(model.keys()):
        if not (is_unet_key(key) or is_clip_key(key)):
            del model[key]
            print(f"-- pruned {key}")
        
    return model

def scale_model(a,weight,skip=[],skip_vae=False,skip_clip=False,small=True,prune=False,**kwargs):
    dtype = torch.float16 if small else torch.float32
    if prune: a = prune_model(a)

    for key in tqdm(a.keys(), desc=f"scaling the weights by {weight}"):
        if "vae" in skip and is_vae_key(key):
            print(f"skipping VAE: {key}")
            continue
        if "clip" in skip and is_clip_key(key):
            print(f"skipping clip: {key}")
            continue
        if "unet" in skip and is_unet_key(key):
            print(f"skipping unet: {key}")
            continue
        
        a[key] = (a[key]*weight).to(dtype)

    print("Done!")

    return a
    
def weighted_sum(models,s,skip=[],skip_vae=False,skip_clip=False,small=True,verbose=False,prune=True,**kwargs):
    dtype = torch.float16 if small else torch.float32
    a,a_name = loadModelWeights(models[0])
    alpha = 0
    
    if prune: a = prune_model(a)

    for i in range(len(models)-1):
        b,b_name = loadModelWeights(models[i+1])
        for key in tqdm(a.keys(), desc=f"{i+1}/{len(models)-1} Performing weighted sum merge between {a_name} and {b_name}"):
            if "vae" in skip and is_vae_key(key):
                print(f"skipping VAE: {key}")
                continue
            if "clip" in skip and is_clip_key(key):
                print(f"skipping clip: {key}")
                continue
            if "unet" in skip and is_unet_key(key):
                print(f"skipping unet: {key}")
                continue
            
            if key in a and key in b:

                if a[key].shape!=b[key].shape or len(a[key].shape)==0: continue
                
                scale = (s[0] if len(s)==1 else s[i])
                alpha = get_alpha(a[key].float().cuda(),b[key].float().cuda(),scale,verbose)
                a[key] = ((1-alpha)*a[key]+(alpha*b[key])).to(dtype)
        a_name = f"{1-alpha}({a_name})+{alpha}({b_name})"

    print("Done!")

    return a, a_name

def maxmerge_unet(models,s,skip=[],skip_vae=False,skip_clip=False,small=True,verbose=False,prune=True,**kwargs):

    assert (s is None) or (len(s)>=1), "only a single scaling factor can be used in max merge"
    assert len(models)>=2, "must have at least 2 models for max merge"
    dtype = torch.float16 if small else torch.float32
    models = [loadModelWeights(model) for model in models]
    a,a_name = models[0]
    models = models[1:] #skip first model
    
    if prune: a = prune_model(a)

    for key in tqdm(a.keys(), desc=f"Performing max merge between {a_name} and {len(models)} models"):
        if "vae" in skip and is_vae_key(key):
            print(f"skipping VAE: {key}")
            continue
        if "clip" in skip and is_clip_key(key):
            print(f"skipping clip: {key}")
            continue
        if "unet" in skip and is_unet_key(key):
            print(f"skipping unet: {key}")
            continue
        if "model" in key:
            orig_shape = a[key].shape
            a_values = a[key].flatten()
            model_values = torch.stack([m[key].flatten() for (m,_) in models if key in m and m[key].shape==orig_shape], dim=0)
            scale = s[0] if s is not None else 0.5
            alpha = get_alpha(a_values.float().cuda(),model_values.float().cuda(),scale,verbose)
            diffs = torch.abs(model_values*alpha - a_values.unsqueeze(0)*(1-alpha))
            max_diff_idx = torch.argmax(diffs, dim=0)
            a_values = torch.nan_to_num(model_values[max_diff_idx, torch.arange(len(a_values))],nan=0,posinf=1,neginf=-1)
            a[key] = a_values.reshape(orig_shape).to(dtype)

    a_name = "-".join([n for (_,n) in models])
    print("Done!")

    return a, f"{s}-max({a_name})"

def cosine_similarity(models,s,skip=[],skip_vae=False,skip_clip=False,small=True,verbose=False,prune=True,**kwargs):
    # output_file = f'{args.out}-{args.s}.ckpt'
    # step = 0
    
    a,a_name,configs = loadModelWeights(models[0])
    cosine = torch.nn.CosineSimilarity(dim=0)
    dtype = torch.float16 if small else torch.float32
    
    if prune: a = prune_model(a)

    for i in range(len(models)-1):
        b,b_name,_ = loadModelWeights(models[i+1])
        for key in tqdm(a.keys(), desc=f"{i+1}/{len(models)-1} Performing cosine similarity merge between {a_name} and {b_name}"):
            if "vae" in skip and is_vae_key(key):
                print(f"skipping VAE: {key}")
                continue
            if "clip" in skip and is_clip_key(key):
                print(f"skipping clip: {key}")
                continue
            if "unet" in skip and is_unet_key(key):
                print(f"skipping unet: {key}")
                continue
            if key in a and key in b:
                
                if a[key].shape!=b[key].shape or len(a[key].shape)==0: continue
                
                sim_matrix = cosine(a[key].to(torch.float64), b[key].to(torch.float64))
                sim_ab = cosine(a[key].flatten().to(torch.float64), b[key].flatten().to(torch.float64))
                min_sim = sim_matrix.min()
                k = (sim_ab - min_sim)/(sim_matrix.max()-min_sim)
                
                scale = (s[0] if len(s)==1 else s[i])
                alpha = get_alpha(a[key].float().cuda(),b[key].float().cuda(),scale,verbose)
                
                k = torch.nan_to_num(k - alpha,nan=1,posinf=1,neginf=0).clip(min=0.,max=1.)
                
                if (np.isnan(k)): print(sim_ab,min_sim,k)

                a[key] = torch.nan_to_num(a[key] * k + b[key] * (1-k),nan=0,posinf=1,neginf=-1).to(dtype)
        a_name = f"cosine({a_name}+{b_name})"
    print("Done!")

    return a, a_name, configs

def merge(models,s,method,extension,out,weight,**kwargs):

    assert len(models)>=1, "must provide at least 1 model"
    scales = [float(scale.strip()) for scale in s.split(",")] if s else get_scales(len(models))
    assert scales is None or len(scales)==1 or len(scales)==len(models)-1, "scale must be blank, 1, or 1 less than number of models"
    assert weight!=0, "weight cannot be 0"

    if method=="sum":
        output_model, out_name = weighted_sum(models,scales,**kwargs)    
    elif method=="cosine":
        output_model, out_name, configs = cosine_similarity(models,scales,**kwargs)
    elif method=="max":
        output_model, out_name = maxmerge_unet(models,scales,**kwargs)
    else:
        output_model, out_name = loadModelWeights(models[0])
        dtype = torch.float16 if kwargs.get("small") else torch.float32
        for key in tqdm(output_model.keys(), desc=f"Converting model dtype to {dtype}"):
            output_model[key] = output_model[key].to(dtype)
        out_name = f"{out_name}-{dtype}"

    output_modelname = f"{out}.{extension}" if out else f"{out_name}.{extension}"
    
    if weight!=1.0:
        output_model = scale_model(output_model,weight,**kwargs)

    if extension.lower() == "safetensors":
        st_torch.save_file(output_model, output_modelname, metadata={"format": "pt"})
    else:
        model = {
            "weight": output_model
        }
        for key,value in configs:
            model[key] = value
        torch.save(model, output_modelname)

    print(f"Checkpoint saved to {output_modelname}.")

if __name__ == "__main__":
    # script mode
    merge(**vars(args))