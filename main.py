from __future__ import print_function

#%matplotlib inline
import random
import time
import torch
import os
from IPython.display import HTML
from torch.cuda import check_error
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
import matplotlib.animation as animation
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

import models
import optimizers

DEFAULTS = {
    # Root directory for dataset
    "dataroot": "data/cifar10",
    # Number of workers for dataloader
    "workers": 1,
    # Batch size during training
    "batch_size": 4,
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    "image_size": 64,
    # Number of channels in the training images. For color images this is 3
    "nc": 1,
    # Size of z latent vector (i.e. size of generator input)
    "nz": 512,
    # Size of feature maps in generator
    "ngf": 32,
    # Size of feature maps in discriminator
    "ndf": 32,
    # Number of training epochs
    "num_epochs": 1,
    # Learning rate for optimizers
    "lr": 2e-4,
    # Beta1 hyperparam for Adam optimizers
    "beta1": 0.5,
    # Number of GPUs available. Use 0 for CPU mode.
    "ngpu": 1,
}

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and DEFAULTS["ngpu"] > 0 else "cpu"
)


def create_dataset(which_dataset, show=False, subset=None):
    match which_dataset:
        case "mnist":
            # Then load the dataset as usual
            dataset = torchvision.datasets.MNIST(
                DEFAULTS["dataroot"],
                transform=transforms.Compose(
                    [
                        transforms.Resize(DEFAULTS["image_size"]),
                        transforms.CenterCrop(DEFAULTS["image_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize(0, 1),
                    ]
                ),
                download=True,
            )
        case _: # Default dataset is cifar10
            dataset = torchvision.datasets.CIFAR10(
                DEFAULTS["dataroot"],
                transform=transforms.Compose(
                    [
                        transforms.Resize(DEFAULTS["image_size"]),
                        transforms.CenterCrop(DEFAULTS["image_size"]),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
                download=True,
            )
    if subset is not None:
        dataset = torch.utils.data.Subset(dataset, subset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=DEFAULTS["batch_size"],
        shuffle=True,
        num_workers=DEFAULTS["workers"],
    )
    if show:
        features, labels = next(iter(dataloader))
        print(f"Feature batch shape: {features.size()}")
        print(f"Labels batch shape: {labels.size()}")
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("haha")
        plt.imshow(
            np.transpose(
                vutils.make_grid(
                    features.to(DEVICE)[:64], padding=2, normalize=True
                ).cpu(),
                (1, 2, 0),
            )
        )
        plt.show()
    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--nepochs",
        type=int,
        default=DEFAULTS["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
            "-d",
            "--dataset",
            type=str,
            default="cifar10",
            help="What dataset to use"
            )
    args = parser.parse_args()
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader = create_dataset(args.dataset, subset=range(1000))

    ngpu = DEFAULTS["ngpu"]

    generator = models.Generator(ngpu).to(DEVICE)

    # Handle multi-gpu setup
    if DEVICE.type == "cuda" and ngpu > 1:
        generator = nn.parallel.DataParallel(generator, list(range(ngpu)))

    # Initialize the weights
    generator.apply(models.weights_init)

    print(generator)

    # Create the Discriminator
    discriminator = models.Discriminator(ngpu).to(DEVICE)

    # Handle multi-gpu if desired
    if (DEVICE.type == "cuda") and (ngpu > 1):
        discriminator = nn.parallel.DataParallel(discriminator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator.apply(models.weights_init)

    # Print the model
    print(discriminator)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, DEFAULTS["nz"], 1, 1, device=DEVICE)

    real_label = 1.0
    fake_label = 0.0

    # opti_d = optim.Adam(
    #     discriminator.parameters(), lr=DEFAULTS["lr"], betas=(DEFAULTS["beta1"], 0.999)
    # )
    # opti_g = optim.Adam(
    #     generator.parameters(), lr=DEFAULTS["lr"], betas=(DEFAULTS["beta1"], 0.999)
    # )

    opti_d = optimizers.ExtraSGD(discriminator.parameters(), lr=DEFAULTS["lr"])
    opti_g = optimizers.ExtraSGD(generator.parameters(), lr=DEFAULTS["lr"])

    # Training loop
    img_list = []
    d_losses = []
    g_losses = []
    iters = 0

    for epoch in range(args.nepochs):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            # Train with all-real batch
            discriminator.zero_grad()
            real = data[0].to(DEVICE)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=DEVICE)

            output_d = discriminator(real).view(-1)
            err_d_real = criterion(output_d, label)
            err_d_real.backward()
            d_x = output_d.mean().item()

            # Train with all-fake batch
            noise = torch.randn(b_size, DEFAULTS["nz"], 1, 1, device=DEVICE)
            fake = generator(noise)
            label.fill_(fake_label)
            output_d = discriminator(fake.detach()).view(-1)
            err_d_fake = criterion(output_d, label)
            err_d_fake.backward()
            d_g_z1 = output_d.mean().item()
            err_d = err_d_real + err_d_fake

            opti_d.extrapolate()
            opti_d.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ############################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake
            # batch through D
            output = discriminator(fake).view(-1)
            err_g = criterion(output, label)
            err_g.backward()
            d_g_z2 = output.mean().item()

            opti_g.extrapolate()
            opti_g.step()

            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        args.nepochs,
                        i,
                        len(dataloader),
                        err_d.item(),
                        err_g.item(),
                        d_x,
                        d_g_z1,
                        d_g_z2,
                    )
                )
            g_losses.append(err_g.item())
            d_losses.append(err_d.item())

            if (iters % 500 == 0) or (
                (epoch == DEFAULTS["num_epochs"] - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        # At the end of an epoch we save the two networks
        checkpoints_path = "model_checkpoints/"
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)
        timestamp = time.strftime("%b-%d-%Y_%H%M", time.localtime())
        filename = lambda x: checkpoints_path + x + "-" + str(epoch) + "-" + args.dataset + "-" + timestamp + ".checkpoint"
        torch.save(discriminator, filename("discriminator"))
        torch.save(generator, filename("generator"))


    timestamp = time.strftime("%b-%d-%Y_%H%M", time.localtime())

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("losses/losses-" + timestamp + ".png")

    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(
        fig, ims, interval=1000, repeat_delay=1000, blit=True
    )

    HTML(ani.to_jshtml())

    # Grab a batch of real images from the dataloader
    real_batch, _ = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(
        vutils.make_grid(
            real_batch.to(DEVICE)[:64], padding=2, normalize=True
        ).cpu(),(1,2,0))
    )

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.savefig("generated/generated-" + timestamp + ".png")


if __name__ == "__main__":
    main()
