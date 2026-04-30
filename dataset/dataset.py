import torch
from torch.utils.data import Dataset
import pickle
import pandas as pd
import numpy as np
import os
from decord import VideoReader
from decord import cpu, gpu
from PIL import Image
import random

# Dataset only for training - we only extract the tubelets
class FSARMultisportsDatasetTrain(Dataset):
    def __init__(self, data_root, gt_path, num_frames, gpu_available=False, transform=None):
        self.data_root = data_root
        self.gt_path = gt_path
        self.transform = transform
        self.gpu_available = gpu_available
        self.num_frames = num_frames
        

        # Load gt data
        with open(gt_path, "rb") as file:
            data = pickle.load(file)

        # To get the labels from label ids
        self.label_dict = data["train_labels"]

        # Train videos
        train_videos_set = set(data["train_videos"])

        # Get the training samples
        self.samples = []
        for video_id, gttube_video_dicts in data["gttubes"].items():

            if video_id not in train_videos_set:
                continue

            for label, gttubes_video in gttube_video_dicts.items():
                for gttube in gttubes_video:
                    gttube = np.array(gttube)
                    self.samples.append({
                        "video_id": video_id,
                        "label": label,
                        "frame_indices": gttube[:, 0].astype(int).tolist(),
                        "bboxes": gttube[:, 1:]
                    })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample["video_id"]
        frame_indices = sample["frame_indices"]
        bboxes = sample["bboxes"]
        label = sample["label"]

        # Uniformly sample frames and bboxes
        tublet_num_frames = len(frame_indices)
        sampled_idxs = np.linspace(0, tublet_num_frames -1, self.num_frames).astype(int)
        frame_indices = [frame_indices[i] - 1 for i in sampled_idxs] # The frame annotations are from 1
        bboxes = [bboxes[i] for i in sampled_idxs]


        # Construct path for video
        video_path = os.path.join(self.data_root, f"{video_id}.mp4")
        # Read video
        if self.gpu_available:
            vr = VideoReader(video_path, ctx=gpu(0))
        else:
            vr = VideoReader(video_path, ctx=cpu(0))

        video_data = vr.get_batch(frame_indices).asnumpy()

        del vr

        # Get tublets
        tublet_frames = []
        for i, frame in enumerate(video_data):
            # Convert to PIL to be able to crop
            img = Image.fromarray(frame)

            # Crop
            bbox = bboxes[i]
            cropped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

            if self.transform:
                cropped_img = self.transform(cropped_img)

            tublet_frames.append(cropped_img)

        # Convert to tensor, TODO change for specific model
        tublet_tensor = torch.stack(tublet_frames).permute(1, 0, 2, 3)
        return tublet_tensor, torch.tensor(label)


class 