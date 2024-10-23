import sys

sys.path.append("..")

import numpy as np
from tqdm.auto import tqdm, trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import cv2

from model import *
from utils import *
from trainer import Trainer

cap = cv2.VideoCapture("dron_0_20240718_134752_047E-44-65.mkv")

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

model = DAIFNet(4, 4)
model.cuda()
model_dict = torch.load("best-model.pth")
state_dict = model_dict["model"]
optim_dict = model_dict["optimizer"]
model.load_state_dict(state_dict)

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.resize(frame, (2048, 1024))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(np.float32) / 255.0
        # Create a dummy 4th channel (zeros with the same height and width)
        dummy_channel = np.zeros(
            (image_rgb.shape[0], image_rgb.shape[1], 1), dtype=np.float32
        )

        # Concatenate the dummy channel to the image, making it 4-channel (H, W, 4)
        image_4ch = np.concatenate((image_rgb, dummy_channel), axis=2)

        # Convert to PyTorch tensor and permute to match the input dimensions (C, H, W)
        image_tensor = (
            torch.tensor(image_4ch).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        )  # Shape: (1, 1, 4, 1024, 2048)
        image_tensor = image_tensor.to("cuda")
        output = model(image_tensor)

        # Assuming the output tensor is in shape (1, C, H, W)
        output = output.squeeze(0)  # Remove the batch dimension, now shape (C, H, W)
        raw_aif = output[:, :-1]
        raw_dpt = output[:, -1]
        # print(raw_dpt.min(),raw_dpt.max())

        # Convert the tensor from PyTorch to NumPy
        output_np = output.permute(1, 2, 0).cpu().detach().numpy()  # Shape: (H, W, C)
        dpt = output_np[:, :, 3]
        dpt -= dpt.min(axis=(0, 1))
        dpt /= dpt.max(axis=(0, 1))

        # If the output is in the range [0, 1], convert it to [0, 255]
        output_np = (output_np * 255).astype(np.uint8)
        dpt = (dpt * 255).astype(np.uint8)

        # If you need to convert from RGB back to BGR for OpenCV
        aif = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        # Display the resulting frame
        cv2.imshow("Frame", aif)
        cv2.imshow("Last channel", cv2.applyColorMap(dpt, cv2.COLORMAP_RAINBOW))

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
