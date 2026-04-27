import torch

class Config:

    # Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 