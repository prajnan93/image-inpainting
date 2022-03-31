def random_ff_mask():
    """
    Free form mask

    """
    raise NotImplementedError()


def random_bbox_mask():
    """
    Bounding box mask

    """
    raise NotImplementedError()


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
