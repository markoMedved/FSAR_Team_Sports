from torch.utils.data import DataLoader
from .dataset import FSARMultisportsDatasetTrain



def build_train_dataloader(cfg):

    # build dataset
    ds = FSARMultisportsDatasetTrain(data_root=cfg.data_root, 
                                     gt_path=cfg.gt_path, 
                                     gpu_available=cfg.device == "cuda", 
                                     transform=cfg.transforms,
                                     num_frames=cfg.num_frames)
    
    return DataLoader(dataset=ds, 
                      batch_size=cfg.batch_size, 
                      shuffle=True,
                      num_workers=cfg.num_workers)

