import argparse
import warnings

from torch.utils.data import DataLoader

from inpaint.core.discriminator import PatchDiscriminator
from inpaint.core.generator import GatedGenerator
from inpaint.data import PlacesDataset
from inpaint.tools import Trainer

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="Train a GAN model for inpainting")

    # Data args
    parser.add_argument(
        "--train_ds_dir",
        type=str,
        help="Path of root directory for the training dataset",
    )
    parser.add_argument(
        "--val_ds_dir",
        type=str,
        help="Path of root directory for the training dataset",
    )
    parser.add_argument(
        "--LOG_DIR",
        type=str,
        required=True,
        help="Path of the directory where logs are to be written",
    )
    parser.add_argument(
        "--CKPT_DIR",
        type=str,
        required=True,
        help="Path of the directory where checkpoints are to be saved",
    )
    parser.add_argument(
        "--SAMPLE_DIR",
        type=str,
        required=True,
        help="Path of the directory where Sample Images are to be saved",
    )

    # Image and Mask args
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=True,
        help="free_form or box mask type",
    )
    parser.add_argument("--mask_num", type=int, default=20, help="Number of masks")
    parser.add_argument(
        "--max_angle", type=int, default=10, help="max angle of free form mask"
    )
    parser.add_argument(
        "--max_len", type=int, default=40, help="max length of free form mask"
    )
    parser.add_argument(
        "--max_width", type=int, default=50, help="max width of free form mask"
    )
    parser.add_argument(
        "--margin", type=int, nargs="+", default=None, help="margin for box mask"
    )
    parser.add_argument(
        "--bbox_shape", type=int, nargs="+", default=None, help="shape of box mask"
    )

    # Training args
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=10000,
        help="Number of steps per epochs to train for",
    )
    parser.add_argument(
        "--val_steps", type=int, default=1000, help="Number of validation steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Training batch size"
    )
    parser.add_argument("--lr_g", type=float, default=2e-4, help="Adam: learning rate")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="Adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="Adam: beta 1")
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: beta 2")
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="Adam: weight decay"
    )
    parser.add_argument(
        "--lr_decrease_epoch",
        type=int,
        default=10,
        help="lr decrease at certain epoch and its multiple",
    )
    parser.add_argument(
        "--lr_decrease_factor",
        type=float,
        default=0.5,
        help="lr decrease factor, for classification default 0.1",
    )
    parser.add_argument(
        "--lambda_l1", type=float, default=100, help="the parameter of L1Loss"
    )
    parser.add_argument(
        "--lambda_perceptual",
        type=float,
        default=10,
        help="the parameter of FML1Loss (perceptual loss)",
    )
    parser.add_argument(
        "--lambda_gan",
        type=float,
        default=1,
        help="the parameter of valid loss of AdaReconL1Loss; 0 is recommended",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument("--device_id", type=int, default=0, help="gpu device id")
    parser.add_argument("--use_cuda", action="store_true", help="Enable scheduler")
    parser.add_argument("--LOG_INTERVAL", type=int, default=100, help="log interval")
    parser.add_argument(
        "--SAVE_SAMPLES_INTERVAL", type=int, default=5, help="sample interval"
    )
    parser.add_argument(
        "--SAVE_SAMPLE_COUNT", type=int, default=5, help="number of samples to save"
    )

    # Model args
    parser.add_argument(
        "--in_channels", type=int, default=4, help="input RGB image + 1 channel mask"
    )
    parser.add_argument("--out_channels", type=int, default=3, help="output RGB image")
    parser.add_argument(
        "--latent_channels", type=int, default=48, help="latent channels"
    )
    parser.add_argument("--pad_type", type=str, default="zero", help="the padding type")
    parser.add_argument(
        "--activation", type=str, default="lrelu", help="the activation type"
    )
    parser.add_argument(
        "--norm_d",
        type=str,
        default="none",
        help="normalization type for discriminator",
    )
    parser.add_argument(
        "--norm_g",
        type=str,
        default="instance",
        help="normalization type for generator",
    )
    parser.add_argument(
        "--init_type", type=str, default="xavier", help="the initialization type"
    )
    parser.add_argument(
        "--init_gain", type=float, default=0.02, help="the initialization gain"
    )
    parser.add_argument(
        "--use_perceptualnet", action="store_true", help="Enable scheduler"
    )
    parser.add_argument(
        "--sn_enable", action="store_true", help="Enable spectral normalisation"
    )

    cfg = parser.parse_args()

    # Initialize Train and Validation loader
    train_ds = PlacesDataset(
        path_dir=cfg.train_ds_dir,
        transform_config=("to_tensor", "random_crop", "norm"),
        crop_size=cfg.crop_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    print(f"Total training images: {len(train_ds)}")

    val_ds = PlacesDataset(
        path_dir=cfg.val_ds_dir,
        transform_config=("to_tensor", "center_crop", "norm"),
        crop_size=cfg.crop_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )
    print(f"Total validation images: {len(val_ds)}")

    # Initialize Discriminator and Generator models
    generator = GatedGenerator(cfg)
    discriminator = PatchDiscriminator(cfg)

    # Initialize Trainer
    trainer = Trainer(
        cfg=cfg,
        discriminator=discriminator,
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print("Training!\n")
    trainer.train()

    print("\nTraining Completed!\n")


if __name__ == "__main__":
    main()
