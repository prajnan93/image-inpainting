from copy import deepcopy

import torch


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

    def _train_discriminator(self, optimizer_dis, image):
        raise NotImplementedError

    def _train_generator(self, optimizer_gen):
        raise NotImplementedError

    def _train_gan(self):

        generator = self.generator
        discriminator = self.discriminator

        best_generator = deepcopy(generator)
        best_discriminator = deepcopy(discriminator)

        start_epoch = 0
        total_epochs = self.cfg.epochs

        for epoch in range(start_epoch, start_epoch + total_epochs):
            for iteration, img in enumerate(self.train_loader):

                img = img.to(self.device)

    def train(self):
        raise NotImplementedError

        # TODO: Setup checkpoints and log directory

        # TODO: Save best discriminator and generator model

    def _validate_generator(self):
        raise NotImplementedError

    def _validate_discriminator(self):
        raise NotImplementedError
