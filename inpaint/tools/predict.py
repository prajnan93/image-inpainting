import numpy as np
import torch


def predict(generator, img, mask):
    generator.eval()

    img = torch.from_numpy(img)
    mask = torch.from_numpy(img)  # 1x C x H x W

    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)

    coarse_output, refine_output = generator(img, mask)

    refine_output = refine_output.squeeze()

    return refine_output.cpu().detach().numpy()
