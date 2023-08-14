import torch

import sys

weights = torch.load(sys.argv[1])

new_weights = {}
for k, v in weights.items():
    if k.startswith("embedder.encoder.embeddings."):
        new_weights[k.replace("embedder.encoder.embeddings.", "embeddings.")] = v
    elif k.startswith("embedder.encoder."):
        new_weights[k.replace("embedder.encoder.", "")] = v
    else:
        raise ValueError(f"weight name not supported:{k}")

torch.save(new_weights, sys.argv[1])

