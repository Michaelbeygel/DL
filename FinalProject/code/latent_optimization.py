import torch
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
clip_loss = utils.CLIPLoss(device=device, dtype=dtype)