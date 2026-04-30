import torch
from dataset.transforms import get_dinov2_transforms

class Config:

    # Device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    data_root = "multisports_data/data/trainval"
    gt_path="multisports_fewshot_GT.pkl"

    # Training parameters
    batch_size = 16
    num_workers = 0
    transform = get_dinov2_transforms()

    # Amount of frames used 
    num_frames = 16