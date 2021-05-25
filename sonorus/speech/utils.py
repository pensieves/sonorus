import torch


def to_device(model, gpu_idx, for_eval=True):

    if model is not None:

        if gpu_idx is not None and torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_idx}")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        if for_eval:
            model = model.eval()

    return model
