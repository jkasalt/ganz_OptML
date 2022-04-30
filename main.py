from __future__ import print_function

#%matplotlib inline
import random
from sys import settrace
from matplotlib.fontconfig_pattern import generate_fontconfig_pattern
import torch
from torch._C import set_num_interop_threads
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import dataloader
import torchvision.utils as vutils
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import models

SETTINGS = {
    # Root directory for dataset
    "dataroot": "data/cifar10",
    # Number of workers for dataloader
    "workers": 2,
    # Batch size during training
    "batch_size": 128,
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    "image_size": 64,
    # Number of channels in the training images. For color images this is 3
    "nc": 3,
    # Size of z latent vector (i.e. size of generator input)
    "nz": 100,
    # Size of feature maps in generator
    "ngf": 64,
    # Size of feature maps in discriminator
    "ndf": 64,
    # Number of training epochs
    "num_epochs": 5,
    # Learning rate for optimizers
    "lr": 2e-4,
    # Beta1 hyperparam for Adam optimizers
    "beta1": 0.5,
    # Number of GPUs available. Use 0 for CPU mode.
    "ngpu": 1,
    # Set gpu or cpu
}

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() and SETTINGS["ngpu"] > 0 else "cpu"
)


def create_dataset(show=False):
    dataset = torchvision.datasets.CIFAR10(
        SETTINGS["dataroot"],
        transform=transforms.Compose(
            [
                transforms.Resize(SETTINGS["image_size"]),
                transforms.CenterCrop(SETTINGS["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        download=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=SETTINGS["batch_size"],
        shuffle=True,
        num_workers=SETTINGS["workers"],
    )
    features, labels = next(iter(dataloader))
    print(f"Feature batch shape: {features.size()}")
    print(f"Labels batch shape: {labels.size()}")
    if show:
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
    seed = 123
    random.seed(seed)
    torch.manual_seed(seed)
    dataloader = create_dataset()

    ngpu = SETTINGS["ngpu"]

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
        netD = nn.parallel.DataParallel(discriminator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator.apply(models.weights_init)

    # Print the model
    print(discriminator)

    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, SETTINGS["nz"], 1, 1, device=DEVICE)

    real_label = 1.0
    fake_label = 0.0

    opti_d = optim.Adam(
        discriminator.parameters(), lr=SETTINGS["lr"], betas=(SETTINGS["beta1"], 0.999)
    )
    opti_g = optim.Adam(
        generator.parameters(), lr=SETTINGS["lr"], betas=(SETTINGS["beta1"], 0.999)
    )

    # Training loop
    img_list = []
    d_losses = []
    g_losses = []
    iters = 0

    for epoch in range(SETTINGS["num_epochs"]):
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
            noise = torch.randn(b_size, SETTINGS["nz"], 1, 1, device=DEVICE)
            fake = generator(noise)
            label.fill_(fake_label)
            output_d = discriminator(fake.detach()).view(-1)
            err_d_fake = criterion(output_d, label)
            err_d_fake.backward()
            d_g_z1 = output_d.mean().item()
            err_d = err_d_real + err_d_fake
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
            opti_g.step()

            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        SETTINGS["num_epochs"],
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
                (epoch == SETTINGS["num_epochs"] - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


if __name__ == "__main__":
    main()
