
import hashlib
import os

def get_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(- 10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except:
        model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

    return model_hash

def load_hubert(config):
    try:
        from fairseq import checkpoint_utils
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
            [os.path.join(os.getcwd(),"models","hubert_base.pt")],
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