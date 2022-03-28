import argparse

from inpaint.tools import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a GAN model for inpainting")

    parser.add_argument(
        "--train_ds",
        type=str,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--train_ds_dir",
        type=str,
        help="Path of root directory for the training dataset",
    )
    parser.add_argument(
        "--val_ds",
        type=str,
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--train_crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size for training images",
    )
    parser.add_argument(
        "--val_crop_size",
        type=int,
        nargs="+",
        default=None,
        help="Crop size for validation images",
    )
    parser.add_argument(
        "--val_ds_dir",
        type=str,
        help="Path of root directory for the validation dataset",
    )
    parser.add_argument(
        "--log_dis_dir",
        type=str,
        required=True,
        help="Path of the directory where logs of the Discriminator are to be written",
    )
    parser.add_argument(
        "--log_gen_dir",
        type=str,
        required=True,
        help="Path of the directory where logs of the Generator are to be written",
    )
    parser.add_argument(
        "--ckpt_dis_dir",
        type=str,
        required=True,
        help="Path of the directory where Discriminator checkpoints are to be saved",
    )
    parser.add_argument(
        "--ckpt_dis_dir",
        type=str,
        required=True,
        help="Path of the directory where Generator checkpoints are to be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, required=False, help="Learning rate")

    args = parser.parse_args()

    # TODO 1: Initialize Train and Validation loader
    train_loader = None
    val_loader = None

    # TODO 2: Initialize Discriminator and Generator models
    discriminator = None
    generator = None

    trainer = Trainer(
        cfg=args,
        discriminator=discriminator,
        generator=generator,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    print("Training")
    trainer.train()


if __name__ == "__main__":
    main()
