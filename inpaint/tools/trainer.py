import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from inpaint.core.modules import PerceptualNet
from inpaint.utils import (
    AverageMeter,
    random_bbox_mask,
    random_ff_mask,
    save_sample_png,
)


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
        train_loader,
        val_loader,
    ):

        self.cfg = cfg

        self.discriminator = discriminator
        self.generator = generator
        self.perceptual_net = None

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

        if self.cfg.use_perceptualnet:
            self.perceptual_net = PerceptualNet()
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

    def _adjust_learning_rate(self, lr_in, optimizer, epoch):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (
            self.cfg.lr_decrease_factor ** (epoch // self.cfg.lr_decrease_epoch)
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _create_mask(self, img):

        B, C, H, W = img.shape

        mask = torch.empty(B, 1, H, W).cuda()

        # set the same masks for each batch
        for i in range(B):
            if self.cfg.mask_type.lower() == "free_form":
                mask[i] = torch.from_numpy(
                    random_ff_mask(
                        shape=(H, W),
                        max_angle=self.cfg.max_angle,
                        max_len=self.cfg.max_len,
                        max_width=self.cfg.max_width,
                        times=self.cfg.mask_num,
                    ).astype(np.float32)
                )
            else:
                mask[i] = torch.from_numpy(
                    random_bbox_mask(
                        shape=(H, W),
                        margin=self.cfg.margin,
                        bbox_shape=self.cfg.bbox_shape,
                        times=self.cfg.mask_num,
                    ).astype(np.float32)
                )

        mask = mask.to(self.device)
        return mask

    def _train_discriminator(self, real_img, mask, optimizer_d):
        B, C, H, W = real_img.shape

        # LSGAN vectors
        valid = torch.FloatTensor(np.ones((B, 1, H // 32, W // 32))).to(self.device)
        zero = torch.FloatTensor(np.zeros((B, 1, H // 32, W // 32))).to(self.device)

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
        preds = self.discriminator(refine_out_wholeimg, mask)

        # GAN Loss
        loss_g = -torch.mean(preds)

        # L1 Reconstruction Loss
        coarse_L1Loss = (coarse_out - img).abs().mean()
        refine_L1Loss = (refine_out - img).abs().mean()
        loss_r = self.cfg.lambda_l1 * coarse_L1Loss + self.cfg.lambda_l1 * refine_L1Loss

        loss_perceptual = 0
        if self.cfg.use_perceptualnet:
            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_feature_maps = self.perceptual_net(img)
            refine_out_feature_maps = self.perceptual_net(refine_out)
            loss_perceptual = F.l1_loss(refine_out_feature_maps, img_feature_maps)

        # Compute overall loss
        loss_total = (
            loss_r
            + self.cfg.lambda_perceptual * loss_perceptual
            + self.cfg.lambda_gan * loss_g
        )

        # Update generator weights
        loss_total.backward()
        optimizer_g.step()

        return (loss_g, loss_r, loss_total)

    def _train_gan(self):

        writer = SummaryWriter(log_dir=self.cfg.LOG_DIR)

        best_generator = deepcopy(self.generator)
        best_discriminator = deepcopy(self.discriminator)

        self.discriminator.train()
        self.generator.train()

        # Setup Optimizers
        optimizer_d, optimizer_g = self._setup_trainer()

        start_epoch = 0
        total_epochs = self.cfg.epochs

        min_avg_val_loss_r = float("inf")
        min_avg_val_loss_d = float("inf")
        min_avg_val_loss_g = float("inf")
        min_avg_val_loss_whole = float("inf")

        # Start Training
        for epoch in range(start_epoch, start_epoch + total_epochs):

            print(f"Epoch {epoch + 1} of {start_epoch + total_epochs}")
            print(120 * "-")

            losses = {
                "loss_g": AverageMeter(),
                "loss_d": AverageMeter(),
                "loss_r": AverageMeter(),
                "loss_total": AverageMeter(),
            }

            for iteration, img in enumerate(self.train_loader):

                img = img.to(self.device)
                B, C, H, W = img.shape

                mask = self._create_mask(img)

                # Train Discriminator
                (
                    loss_d,
                    coarse_out,
                    refine_out,
                    refine_out_wholeimg,
                ) = self._train_discriminator(img, mask, optimizer_d)

                # Train Generator
                loss_g, loss_r, loss_total = self._train_generator(
                    img, mask, coarse_out, refine_out, refine_out_wholeimg, optimizer_g
                )

                # Track loss history
                losses["loss_d"].update(loss_d.item(), B)
                losses["loss_g"].update(loss_g.item(), B)
                losses["loss_r"].update(loss_r.item(), B)
                losses["loss_total"].update(loss_total.item(), B)

                # Logging and Tensorboard Summary Writer
                if iteration % self.cfg.LOG_INTERVAL == 0:
                    total_iterations = iteration + (epoch * len(self.train_loader))

                    print(
                        f"Iteration {iteration}/{total_iterations}"
                        + f" Discriminator Loss: {losses['loss_d'].avg},"
                        + f" GAN Loss: {losses['loss_g'].avg},"
                        + f" Reconstruction Loss: {losses['loss_r'].avg},"
                        + f" Overall Generator Loss: {losses['loss_total'].avg}"
                    )

                    writer.add_scalar(
                        "avg_train_discriminator_loss",
                        losses["loss_d"].avg,
                        total_iterations,
                    )
                    writer.add_scalar(
                        "avg_train_gan_loss", losses["loss_g"].avg, total_iterations
                    )
                    writer.add_scalar(
                        "avg_train_reconstruction_loss",
                        losses["loss_r"].avg,
                        total_iterations,
                    )
                    writer.add_scalar(
                        "avg_train_generator_loss",
                        losses["loss_total"].avg,
                        total_iterations,
                    )

            # Validate and save best model
            val_losses = self._validate_gan(epoch, writer)

            save_best_models = False
            if val_losses["loss_d"].avg < min_avg_val_loss_d:
                min_avg_val_loss_d = val_losses["loss_d"].avg
                save_best_models = True
                print("New avg validation loss discriminator!")

            if val_losses["loss_g"].avg < min_avg_val_loss_g:
                min_avg_val_loss_g = val_losses["loss_g"].avg
                save_best_models = True
                print("New avg validation loss generator!")

            if val_losses["loss_r"].avg < min_avg_val_loss_r:
                min_avg_val_loss_r = val_losses["loss_r"].avg
                save_best_models = True
                print("New avg validation loss reconstruction!")

            if val_losses["loss_total"].avg < min_avg_val_loss_whole:
                min_avg_val_loss_whole = val_losses["loss_total"].avg
                save_best_models = True
                print("New avg validation loss overall!")

            if save_best_models:
                best_discriminator = deepcopy(self.discriminator)
                best_generator = deepcopy(self.generator)

                best_models_dict = {
                    "generator_state_dict": best_generator.state_dict(),
                    "discriminator_state_dict": best_discriminator.state_dict(),
                }

                torch.save(
                    best_models_dict,
                    os.path.join(self.cfg.CKPT_DIR, "best_models.pth"),
                )

                print("Saved best generator and discriminator")

            # Adjust learning rate
            self._adjust_learning_rate(self.cfg.lr_d, optimizer_d, epoch + 1)
            self._adjust_learning_rate(self.cfg.lr_g, optimizer_g, epoch + 1)

            # Save checkpoints
            consolidated_save_dict = {
                "generator_state_dict": self.generator.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "epoch": epoch,
            }
            torch.save(
                consolidated_save_dict,
                os.path.join(
                    self.cfg.CKPT_DIR,
                    "ckpt_epoch" + str(epoch + 1) + ".pth",
                ),
            )

            print("\n")
        
        writer.close()
        return (best_generator, best_discriminator)

    def _validate_gan(self, epoch, writer):

        # Set models to eval state for validation
        self.generator.eval()
        self.discriminator.eval()

        losses = {
            "loss_g": AverageMeter(),
            "loss_d": AverageMeter(),
            "loss_r": AverageMeter(),
            "loss_total": AverageMeter(),
        }

        save_count = 0

        with torch.no_grad():
            for img in self.val_loader:

                img = img.to(self.device)
                B, C, H, W = img.shape

                mask = self._create_mask(img)

                # Discriminator validation
                # LSGAN vectors
                valid = torch.FloatTensor(np.ones((B, 1, H // 32, W // 32))).to(
                    self.device
                )
                zero = torch.FloatTensor(np.zeros((B, 1, H // 32, W // 32))).to(
                    self.device
                )

                # Generate fake pixel values for the given mask and real images
                coarse_out, refine_out = self.generator(img, mask)

                coarse_out_wholeimg = img * (1 - mask) + coarse_out * mask
                refine_out_wholeimg = img * (1 - mask) + refine_out * mask

                # Pass fake images through discriminator
                fake_preds = self.discriminator(refine_out_wholeimg.detach(), mask)
                fake_loss = -torch.mean(torch.min(zero, -valid - fake_preds))

                # Pass real images through discriminator
                real_preds = self.discriminator(img, mask)
                real_loss = -torch.mean(torch.min(zero, -valid + real_preds))

                # Compute overall loss
                loss_d = 0.5 * (fake_loss + real_loss)

                # Generator validation
                # Pass fake image to the discriminaton | Try to fool the discriminator
                preds = self.discriminator(refine_out_wholeimg, mask)

                # GAN Loss
                loss_g = -torch.mean(preds)

                # L1 Reconstruction Loss
                coarse_L1Loss = (coarse_out - img).abs().mean()
                refine_L1Loss = (refine_out - img).abs().mean()
                loss_r = (
                    self.cfg.lambda_l1 * coarse_L1Loss
                    + self.cfg.lambda_l1 * refine_L1Loss
                )

                loss_perceptual = 0
                if self.cfg.use_perceptualnet:
                    # Get the deep semantic feature maps, and compute Perceptual Loss
                    img_feature_maps = self.perceptual_net(img)
                    refine_out_feature_maps = self.perceptual_net(refine_out)
                    loss_perceptual = F.l1_loss(
                        refine_out_feature_maps, img_feature_maps
                    )

                # Compute overall loss
                loss_total = (
                    loss_r
                    + self.cfg.lambda_perceptual * loss_perceptual
                    + self.cfg.lambda_gan * loss_g
                )

                losses["loss_d"].update(loss_d.item(), B)
                losses["loss_g"].update(loss_g.item(), B)
                losses["loss_r"].update(loss_r.item(), B)
                losses["loss_total"].update(loss_total.item(), B)

                # Save intermediate result output
                if (
                    epoch % self.cfg.SAVE_SAMPLES_INTERVAL == 0
                    and save_count < self.cfg.SAVE_SAMPLE_COUNT
                ):
                    save_count += 1

                    masked_img = img * (1 - mask) + mask
                    mask = torch.cat((mask, mask, mask), 1)

                    img_list = [
                        img,
                        mask,
                        masked_img,
                        coarse_out_wholeimg,
                        refine_out_wholeimg,
                    ]
                    name_list = ["gt", "mask", "masked_img", "coarse_out", "refine_out"]

                    sample_path = self.cfg.SAMPLE_DIR + f"/epoch_{epoch+1}"
                    os.makedirs(sample_path, exist_ok=True)
                    save_sample_png(
                        sample_folder=sample_path,
                        sample_name="%d" % (save_count),
                        img_list=img_list,
                        name_list=name_list,
                    )

        print("\nValidation Loss:")
        print(
            f" Discriminator Loss: {losses['loss_d'].avg},"
            + f" GAN Loss: {losses['loss_g'].avg},"
            + f" Reconstruction Loss: {losses['loss_r'].avg},"
            + f" Overall Generator Loss: {losses['loss_total'].avg}"
        )

        writer.add_scalar(
            "avg_val_discriminator_loss",
            losses["loss_d"].avg,
            epoch + 1,
        )
        writer.add_scalar("avg_val_gan_loss", losses["loss_g"].avg, epoch + 1)
        writer.add_scalar(
            "avg_val_reconstruction_loss",
            losses["loss_r"].avg,
            epoch + 1,
        )
        writer.add_scalar(
            "avg_val_generator_loss",
            losses["loss_total"].avg,
            epoch + 1,
        )

        # Set model back to train state after validation
        self.generator.train()
        self.discriminator.train()

        return losses

    def train(self):
        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        os.makedirs(self.cfg.LOG_DIR, exist_ok=True)
        os.makedirs(self.cfg.SAMPLE_DIR, exist_ok=True)

        best_generator, best_discriminator = self._train_gan()

        best_models_dict = {
            "generator_state_dict": best_generator.state_dict(),
            "discriminator_state_dict": best_discriminator.state_dict(),
        }

        torch.save(
            best_models_dict,
            os.path.join(self.cfg.CKPT_DIR, "best_models_final.pth"),
        )

        print("Saved final best generator and discriminator")
