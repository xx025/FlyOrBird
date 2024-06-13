import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_device(batch):
    if isinstance(batch, (list, tuple)):
        return [move_device(data) for data in batch]
    elif isinstance(batch, dict):
        return {key: move_device(data) for key, data in batch.items()}
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
