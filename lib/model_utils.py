
import hashlib

def get_hash(model_path):
    try:
        with open(model_path, 'rb') as f:
            f.seek(- 10000 * 1024, 2)
            model_hash = hashlib.md5(f.read()).hexdigest()
    except:
        model_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

    return model_hash