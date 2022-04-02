import os

import cv2
import numpy as np


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):

        self.reset()

    def reset(self):
        """
        Resets the meter
        """

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the meter
        Parameters
        -----------
        val : float
            Value to update the meter with
        n : int
            Number of samples to update the meter with
        """

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def random_ff_mask(shape, max_angle=10, max_len=40, max_width=50, times=15):
    """Generate a random free form mask with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES,
            VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    height = shape[0]
    width = shape[1]
    mask = np.zeros((height, width), np.float32)
    times = np.random.randint(times - 5, times)
    for i in range(times):
        start_x = np.random.randint(width)
        start_y = np.random.randint(height)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_len - 20, max_len)
            brush_w = 5 + np.random.randint(max_width - 30, max_width)
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return mask.reshape((1,) + mask.shape).astype(np.float32)


def random_bbox(shape, margin, bbox_shape):
    """Generate a random tlhw with configuration.
    Args:
        config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
    Returns:
        tuple: (top, left, height, width)
    """
    img_height = shape[0]
    img_width = shape[1]
    height = bbox_shape[0]
    width = bbox_shape[1]
    ver_margin = margin[0]
    hor_margin = margin[1]
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    return (t, l, h, w)


def random_bbox_mask(shape, margin=(10, 10), bbox_shape=(100, 100), times=15):
    """Generate mask tensor from bbox.
    Args:
        bbox: configuration tuple, (top, left, height, width)
        config: Config should have configuration including IMG_SHAPES,
            MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
    Returns:
        tf.Tensor: output with shape [1, H, W, 1]
    """
    bboxs = []
    for i in range(times):
        bbox = random_bbox(shape, margin, bbox_shape)
        bboxs.append(bbox)
    height = shape[0]
    width = shape[1]
    mask = np.zeros((height, width), np.float32)
    for bbox in bboxs:
        h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
        w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
        mask[
            (bbox[0] + h) : (bbox[0] + bbox[2] - h),
            (bbox[1] + w) : (bbox[1] + bbox[3] - w),
        ] = 1.0
    return mask.reshape((1,) + mask.shape).astype(np.float32)


## for contextual attention


def extract_image_patches(images, ksizes, strides, rates, padding="same"):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ["same", "valid"]
    batch_size, channel, height, width = images.size()

    if padding == "same":
        images = same_padding(images, ksizes, strides, rates)
    elif padding == "valid":
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(
                padding
            )
        )

    unfold = torch.nn.Unfold(
        kernel_size=ksizes, dilation=rates, padding=0, stride=strides
    )
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x
