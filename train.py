import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Generator, Discriminator, initialize_weights

# configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameters
BATCH_SIZE = 128
LR = 2e-4
NUM_EPOCHS = 20
Z_DIM = 100
IMAGE_SIZE = 64
CHANNELS_IMG = 3
FEATURES_DISC = 64
FEATURES_GEN = 64

# model
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
fixed_noise = torch.randn(32,Z_DIM,1,1).to(device)
initialize_weights(gen)
initialize_weights(disc)
transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
    )
])

# dataset
#dataset = datasets.MNIST('dataset/',download=True,train=True,transform=transforms)
dataset = datasets.ImageFolder(root='D:\AI_Workspace\DL_Workspace\Kaggle_Dataset\Celebrity Faces',transform=transforms)
loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

# optimizer & loss
opt_disc = optim.Adam(disc.parameters(),lr=LR, betas=(0.5,0.999))
opt_gen = optim.Adam(gen.parameters(),lr=LR, betas=(0.5,0.999))
loss_fn = nn.BCELoss()

# tensorboard visualization
writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

gen.train()
disc.train()

# training loop
for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1).to(device)
        fake = gen(noise)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = loss_fn(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = loss_fn(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.inference_mode():
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)


            step += 1
            save_gen_img = fake[0]
            save_image(save_gen_img, "images/%d.png" % batch_idx, normalize=True)

torch.save(gen.state_dict(), f='generator_model.pth')