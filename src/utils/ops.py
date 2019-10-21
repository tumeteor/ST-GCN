import sys
import torch


def clear_parameters(model):
    sys.stdout.flush()
    for p in model.parameters():
        if p.grad is not None:
            del p.grad  # free some memory
    torch.cuda.empty_cache()
