import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from inpaint.utils import denorm

def predict(generator, img, mask):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    generator = generator.to(device)
    generator.eval()

    img = torch.from_numpy(img.astype(np.float32))
    mask = torch.from_numpy(mask.astype(np.float32))

    img = img.to(device)
    mask = mask.to(device)

    normalize = transforms.Compose([
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )
    ])

    img = normalize(img)
    
    img = img.unsqueeze(0)
    mask = mask.unsqueeze(0)

    coarse_output, refine_output = generator(img, mask)

    refine_output = denorm(refine_output)

    result = refine_output.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
    result = cv2.convertScaleAbs(result, alpha=(255.0))
    result = np.clip(result, 0, 255)
    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result
