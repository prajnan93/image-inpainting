class Trainer:
    """
    Trainer class for training and validating models.

    Params
    ------

    """

    def __init__(self, cfg, discriminator, generator, train_loader, val_loader):

        self.cfg = cfg

        self.discriminator = discriminator
        self.generator = generator

        self.train_loader = train_loader
        self.val_loader = val_loader

        # TODO: setup models and data for cuda device

    def _discriminator_loss(self):
        raise NotImplementedError

    def _generator_loss(self):
        raise NotImplementedError

    def _train_discriminator(self, optimizer_dis, image):
        raise NotImplementedError

    def _train_generator(self, optimizer_gen):
        raise NotImplementedError

    def _train_gan(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

        # TODO: Setup checkpoints and log directory

        # TODO: Setup loss functions and optimizers

        # TODO: Save best discriminator and generator model

    def _validate_generator(self):
        raise NotImplementedError

    def _validate_discriminator(self):
        raise NotImplementedError
