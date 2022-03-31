from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from inpaint.utils import AverageMeter, random_bbox, random_ff_mask


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
        perceptual_net,
        train_loader,
        val_loader,
    ):

        self.cfg = cfg

        self.discriminator = discriminator
        self.generator = generator
        self.perceptual_net = perceptual_net

        self.train_loader = train_loader
        self.val_loader = val_loader

        self._setup_device()
        self._setup_models()

    def _setup_device(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = torch.device(int(self.cfg.device_id))
            print(f"Running on CUDA device: {self.cfg.device_id}")

        else:
            self.device = torch.device("cpu")
            print("No CUDA device available. Running on CPU.")

    def _setup_models(self):
        self.discriminator = self.discriminator.to(self.device)
        self.generator = self.generator.to(self.device)
        self.perceptual_net = self.perceptual_net.to(self.device)

    def _setup_trainer(self):

        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.cfg.lr_d,
            betas=(self.cfg.b1, self.cfg.b2),
            weight_decay=self.cfg.weight_decay,
        )
        optimzer_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.cfg.lr_g,
            betas=(self.cfg.b1, self.cfg.b2),
            weight_decay=self.cfg.weight_decay,
        )

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
        valid = Tensor(np.ones((B, 1, H // 32, W // 32)))
        zero = Tensor(np.zeros((B, 1, H // 32, W // 32)))

        # Clear the Discriminator gradients
        optimizer_d.zero_grad()

        # Generate fake pixel values for the given mask and real images
        coarse_out, refine_out = self.generator(real_img, mask)

        coarse_out_wholeimg = real_img * (1 - mask) + coarse_out * mask
        refine_out_wholeimg = real_img * (1 - mask) + refine_out * mask

        # Pass fake images through discriminator
        fake_preds = self.discriminator(refine_out_wholeimg.detach(), mask)
        fake_loss = -torch.mean(torch.min(zero, -valid - fake_preds))

        # Pass real images through discriminator
        real_preds = self.discriminator(real_img, mask)
        real_loss = -torch.mean(torch.min(zero, -valid + real_preds))

        # Compute overall loss
        loss_d = 0.5 * (fake_loss + real_loss)

        # Update discriminator weights
        loss_d.backward()
        optimizer_d.step()

        return (loss_d, coarse_out, refine_out, refine_out_wholeimg)

    def _train_generator(
        self, img, mask, coarse_out, refine_out, refine_out_wholeimg, optimizer_g
    ):

        # Clear the Generator gradients
        optimizer_g.zero_grad()

        # Pass fake image to the discriminaton | Try to fool the discriminator
        preds = self.discriminator(refine_out_wholeimg)

        # GAN Loss
        loss_g = -torch.mean(preds)

        # L1 Reconstruction Loss
        coarse_L1Loss = (coarse_out - img).abs().mean()
        refine_L1Loss = (refine_out - img).abs().mean()
        loss_r = self.cfg.lambda_l1 * coarse_L1Loss + self.cfg.lambda_l1 * refine_L1Loss

        # Get the deep semantic feature maps, and compute Perceptual Loss
        img_feature_maps = self.perceptual_net(img)
        refine_out_feature_maps = self.perceptual_net(refine_out)
        loss_perceptual = F.l1_loss(refine_out_feature_map, img_feature_maps)

        # Compute overall loss
        whole_loss = (
            loss_r
            + self.cfg.lambda_perceptual * loss_perceptual
            + self.cfg.lambda_gan * loss_g
        )

        # Update generator weights
        whole_loss.backward()
        optimizer_g.step()

        return (loss_g, loss_r, whole_loss)

    def _train_gan(self):

        writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)

        best_generator = deepcopy(self.generator)
        best_discriminator = deepcopy(self.discriminator)

        self.discriminator.train()
        self.generator.train()

        # Setup Optimizers
        optimizer_d, optimzer_g = self._setup_trainer()

        start_epoch = 0
        total_epochs = self.cfg.epochs

        # Start Training
        for epoch in range(start_epoch, start_epoch + total_epochs):

            print(f"Epoch {epoch + 1} of {start_epoch + total_epochs}")
            print(100 * "-")

            losses = {
                "loss_g": AverageMeter(),
                "loss_d": AverageMeter(),
                "loss_r": AverageMeter(),
                "whole_loss": AverageMeter(),
            }

            for iteration, img in enumerate(self.train_loader):

                img = img.to(self.device)
                mask = self._create_mask(img)

                # Train Discriminator
                (
                    loss_d,
                    coarse_out,
                    refine_out,
                    refine_out_wholeimg,
                ) = self._train_discriminator(img, mask, optimizer_d)

                # Train generator
                loss_g, loss_r, whole_loss = self._train_generator(
                    img, mask, coarse_out, refine_out, refine_out_wholeimg, optimizer_g
                )

                # Track loss history
                losses["loss_d"].update(loss_d.item(), img.size(0))
                losses["loss_g"].update(loss_g.item(), img.size(0))
                losses["loss_r"].update(loss_r.item(), img.size(0))
                losses["whole_loss"].update(whole_loss.item(), item.size(0))

                # Logging and Tensorboard Summary Writer
                if iteration % self.cfg.LOG_INTERVAL == 0:
                    total_iterations = iteration + (epochs * len(self.train_loader))

                    print(
                        f"Iteration {iteration}/{total_iterations}"
                        + f" Discriminator Loss: {losses['loss_d'].avg},"
                        + f" GAN Loss: {losses['loss_g'].avg},"
                        + f" Reconstruction Loss: {losses['loss_r'].avg},"
                        + f" Overall Generator Loss: {losses['whole_loss'].avg}"
                    )

                    writer.add_scaler(
                        "avg_batch_Discriminator_loss",
                        losses["loss_d"].avg,
                        total_iterations,
                    )
                    writer.add_scaler(
                        "avg_batch_GAN_loss", losses["loss_g"].avg, total_iterations
                    )
                    writer.add_scaler(
                        "avg_batch_Reconstruction_loss",
                        losses["loss_r"].avg,
                        total_iterations,
                    )
                    writer.add_scaler(
                        "avg_batch_Generator_loss",
                        losses["whole_loss"].avg,
                        total_iterations,
                    )

                # TODO: Save intermediate result output

    def train(self):
        raise NotImplementedError

        # TODO: Setup checkpoints and log directory

        # TODO: Save best discriminator and generator model

    def _validate_generator(self):
        raise NotImplementedError

    def _validate_discriminator(self):
        raise NotImplementedError
