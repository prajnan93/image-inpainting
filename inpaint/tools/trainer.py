from copy import deepcopy

import torch

from inpaint.utils import random_bbox, random_ff_mask


class Trainer:
    """
    Trainer class for training and validating models.

    Params
    ------

    """

    def __init__(
        self,
        cfg,
        discriminator,
        generator,
        perceptual_network,
        train_loader,
        val_loader,
    ):

        self.cfg = cfg

        self.discriminator = discriminator
        self.generator = generator
        self.perceptual_network = perceptual_network

        self.train_loader = train_loader
        self.val_loader = val_loader

        self._setup_device()
        self._setup_models()

    def _setup_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(int(self.cfg.device))
            print(f"Running on CUDA device: {self.cfg.device}")
        else:
            self.device = torch.device("cpu")
            print("No CUDA device available. Running on CPU.")

    def _setup_models(self):
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)
        self.perceptual_network = self.perceptual_network.to(self.device)

    def _setup_trainer(self):

        # TODO initialize optimizer
        optimizer_d = None
        optimzer_g = None
        return (optimizer_d, optimzer_g)

    def _create_mask(self, img):

        B, C, H, W = img.shape

        if self.cfg.MASK_TYPE.lower() == "free_form":
            # generate free form mask
            generate_mask = random_ff_mask
        else:
            # generate bounding box mask
            generate_mask = random_bbox

        mask = torch.empty(B, 1, H, W).cuda()
        # set the same masks for each batch
        for i in range(opt.batch_size):
            mask[i] = torch.from_numpy(generate_mask(shape=(H, W)).astype(np.float32))

        mask = mask.to(self.device)
        return mask

    def _train_discriminator(self, real_img, mask, optimizer_d):
        B, C, H, W = img.shape

        # LSGAN vectors
        fake = Tensor(np.zeros((B, 1, H // 32, W // 32)))
        valid = Tensor(np.ones((B, 1, H // 32, W // 32)))
        zero = Tensor(np.zeros((B, 1, H // 32, W // 32)))

        raise NotImplementedError

    def _train_generator(self, optimizer_g):
        raise NotImplementedError

    def _train_gan(self):

        best_generator = deepcopy(self.generator)
        best_discriminator = deepcopy(self.discriminator)

        # Setup Optimizers
        optimizer_d, optimzer_g = self._setup_trainer()

        start_epoch = 0
        total_epochs = self.cfg.epochs

        # Start Training
        for epoch in range(start_epoch, start_epoch + total_epochs):
            for iteration, img in enumerate(self.train_loader):

                img = img.to(self.device)
                mask = self._create_mask(img)

                # Train Discriminator
                (
                    loss_d,
                    real_score,
                    fake_score,
                    coarse_out,
                    refine_out,
                    refine_out_wholeimg,
                ) = self._train_discriminator(img, mask, optimizer_d)

                # Train generator
                loss_g = self._train_generator(
                    img, mask, coarse_out, refine_out, refine_out_wholeimg, optimizer_g
                )

                # TODO: Summary writer | Save intermediate result output

    def train(self):
        raise NotImplementedError

        # TODO: Setup checkpoints and log directory

        # TODO: Save best discriminator and generator model

    def _validate_generator(self):
        raise NotImplementedError

    def _validate_discriminator(self):
        raise NotImplementedError
