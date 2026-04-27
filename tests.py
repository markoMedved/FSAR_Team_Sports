from dataset.dataset import FSARMultisportsDatasetTrain
import torch 

# 2. Then import your other librarie
import cv2
import pickle


data_root = "multisports_data/data/trainval"
# video_id = "volleyball/v_0kUtTtmLaJA_c001"

# frames = read_video(video_id)

# frames = crop_video_tubelets(video_id)
# print(frames.keys())


# for i in range(len(frames)):
#     # Note: If crop_video_tubelets returns a list of dicts, 
#     # make sure 'frames' is the list of arrays.
#     frame = frames[i] 

#     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#     # Display the frame index on the window title or on the image
#     cv2.imshow("Video Test - Frame " + str(i), frame_bgr)

#     # waitKey(0) stops the execution until a key is pressed
#     key = cv2.waitKey(0) & 0xFF

#     if key == ord("q"):
#         print("Quitting...")
#         break
#     elif key == ord("d"):
#         # Just continue the loop to the next 'i'
#         continue 
#     else:
#         # If any other key is pressed, stay on this frame index (decrement i)
#         # Or just continue to the next anyway.
#         pass

# cv2.destroyAllWindows()

ground_truth_path="multisports_fewshot_GT.pkl"
# with open(ground_truth_path, "rb") as file:
#     data = pickle.load(file)

# print(data["gttubes"])

# from dataset.transforms import get_dinov2_transforms

# ds  = FSARMultisportsDatasetTrain(data_root=data_root, gt_path=ground_truth_path, transform=get_dinov2_transforms())

# tublet, lab = ds.__getitem__(0)
# print(lab, tublet.shape)

from dataset.dataloader import build_train_dataloader
from config import Config

dataloader = build_train_dataloader(Config())