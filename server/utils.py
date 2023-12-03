def to_response(item: dict, filter: str=""):
    response = {}
    for (k,v) in item.items():
        if filter not in k: continue
        if hasattr(v,"keys"): response[k]=v.keys()
        if hasattr(v,"__dict__"): response[k]=v.__dict__.keys()
        else:
            sv = str(v)
            if len(sv)<32: response[k]=v
            else: response[k]=sv[:32]
    return response