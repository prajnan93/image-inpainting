import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv


# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, "r")
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][: len(content[i]) - 1]
    file.close()
    return content


def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)


def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            if filespath.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
            ):
                ret.append(os.path.join(root, filespath))
    return ret


def get_names(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret


def text_save(content, filename, mode="a"):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + "\n")
    file.close()


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt=255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is sigmoid activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        # Save to certain path
        save_img_name = sample_name + "_" + name_list[i] + ".jpg"
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)
