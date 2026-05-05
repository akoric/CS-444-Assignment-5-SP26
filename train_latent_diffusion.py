import os
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers import AutoencoderKL

from DiT import DiT

# Use cuSolver instead of MAGMA for GPU linalg (required for FID on this cluster)
torch.backends.cuda.preferred_linalg_library("cusolver")

# Hyperparameters ===================================================================================
res = 8 # latent spatial size (64 / 8)
patch_size = 2 # patch size in latent space
channels = 4 # latent channels
heads = 8
hidden_dim = 256
num_patch = (res // patch_size) ** 2  # = 16
ff_dim = 1024
time_emb_dim = 256
num_blocks = 5
num_timesteps = 300
beta_0 = 0.0001
beta_t = 0.02
batch_size = 64
num_epoch = 100
lr = 4e-4

OUTPUT_DIR = 'batch_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Device & VAE ===================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
vae.eval()

# freeze VAE weights
for p in vae.parameters():
    p.requires_grad = False

VAE_SCALE = 0.13025  # SDXL-VAE scaling factor, from Hugging Face

@torch.no_grad()
def encode(x):
    # x: (B, 3, 64, 64) normalized to [-1, 1]
    posterior = vae.encode(x).latent_dist
    z = posterior.sample() * VAE_SCALE
    # z: (B, 4, 8, 8)
    return z

@torch.no_grad()
def decode(z):
    # z: (B, 4, 8, 8)
    z = z / VAE_SCALE
    x = vae.decode(z).sample
    # x: (B, 3, 64, 64)
    return x

# Dataset  ===================================================================================
class CelebADataset(Dataset):
    """Reads directly from img_align_celeba/ folder (Kaggle download)."""
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, 'img_align_celeba', 'img_align_celeba')
        self.transform = transform
        self.imgs = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.imgs[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # dummy label to match CelebA interface


### Helper function to show images  ===================================================================================
def show_images(imgs):
    batch_min = imgs.view(imgs.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    batch_max = imgs.view(imgs.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    imgs = (imgs - batch_min) / (batch_max - batch_min + 1e-6)
    imgs = imgs.detach().cpu()
    grid = make_grid(imgs, nrow=5)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

# LatentDiffusion ===================================================================================

class LatentDiffusion:

    def __init__(self):

        #[DiT-VAE] Resize to 64 x 64 pixel space (not 28x28 MNIST size)
        # [DiT-VAE] CenterCrop make a sq image after resize
        # [DiT-VAE] 3-channel norma instead of single: CelebA is RGB(3)not grayscale(1)
        trans = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        self.num_fid_samples = 1000

        # [DiT-VAE] CelebA (RGB faces) instead of MNIST (grayscale digits)
        self.dataset = CelebADataset(root='./data/celeba', transform=trans)
        self.training_dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        ## Create a linear schedule for the betas
        self.betas = torch.linspace(beta_0, beta_t, num_timesteps).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)

        # Alpha bar (cumulative product)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)

        # [DiT-VAE] num_channels=channels is now 4 (latent channels) instead of 1 (MNIST grayscale)
        #[DiT-VAE] num_patches=num_patch is now 16 (8x8 latent / patch_size=2)^2 instead of 49
        self.model = DiT(
            patch_size=patch_size,
            num_blocks=num_blocks,
            num_heads=heads,
            ff_dim=ff_dim,
            time_emb_dim=time_emb_dim,
            num_timesteps=num_timesteps,
            hidden_dim=hidden_dim,
            num_patches=num_patch,
            num_channels=channels
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.MSELoss()

        os.makedirs('checkpoints', exist_ok=True)
        self._precompute_fid_real()


    def load_checkpoint(self, path='checkpoints/checkpoint_VAE.pt'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from epoch {checkpoint['epoch'] + 1}")
        return (
            checkpoint['epoch'] + 1,
            checkpoint.get('total_losses', []),
            checkpoint.get('all_fids', []),
            checkpoint.get('fid_epochs', []),
        )


    def train(self, start_epoch=0, total_losses=None, all_fids=None, fid_epochs=None):
        total_losses = total_losses or []
        all_fids = all_fids or []
        fid_epochs = fid_epochs or []
        for e in range(start_epoch, num_epoch):
            self.model.train()

            all_losses = []

            for imgs, _ in tqdm(self.training_dataloader):
                # get clean pixel image x0: (B, 3, 64, 64)
                imgs = imgs.to(self.device)

                # [DiT-VAE] encode pixel images to z space
                # VAE downsamples 64x64x3 -> 8x8x4; diffusion runs entirely in latent space
                with torch.no_grad():
                    z = encode(imgs)   # (B, 4, 8, 8)

                # sample a random timestep t
                # [DiT-VAE] z.shape[0] instead of imgs.shape[0] 
                # same batch size is the same but z is the target instead of img
                t = torch.randint(0, num_timesteps, (z.shape[0],), device=self.device)

                ## Create a random noise vector
                eps = torch.randn_like(z)

                ## Run the diffusion process
                ab = self.alpha_bars[t].view(-1, 1, 1, 1)  # (64,) -> (64, 1, 1, 1)
                xt_noise = torch.sqrt(ab) * z + torch.sqrt(1 - ab) * eps

                # Transformer predicts the noise
                noise_pred = self.model(xt_noise, t)

                # Compute the loss and append it to all_losses
                loss = self.loss_fn(noise_pred, eps)
                all_losses.append(loss.item())

                # Backpropagate the loss
                self.optimizer.zero_grad()
                loss.backward()

                ## Update the model parameters
                self.optimizer.step()

            print("Avg Loss:", sum(all_losses) / len(all_losses))
            total_losses += all_losses
            all_losses = []

            # Save checkpoint after every epoch
            torch.save({
                'epoch': e,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_losses': total_losses,
                'all_fids': all_fids,
                'fid_epochs': fid_epochs,
            }, 'checkpoints/checkpoint_VAE.pt')

            if e % 10 == 0:
                print("Epoch:", e + 1)

                images = self.run_inference()
                fid_val = self.compute_fid()
                print("FID:", fid_val)
                all_fids.append(fid_val)
                fid_epochs.append(e + 1)
                show_images(images)

        print("Final Epoch:")
        final_imags = self.run_inference()
        show_images(final_imags)

        plt.plot(total_losses)
        plt.title("Loss")
        plt.xlabel("Steps")
        plt.show()

        plt.plot(fid_epochs, all_fids)
        plt.title("FID")
        plt.xlabel("Epochs")
        plt.show()


    def run_inference(self, num_samples=20):
        '''
        Algorithm 2: DDPM Sampling
        This function is used to generate samples from the model.
        You will generate num_samples samples in total.
        '''
        self.model.eval()
        with torch.no_grad():
            # [DiT-VAE] noise starts in z space (channels=4, res=8) instead of pixel space
            xt = torch.randn(num_samples, channels, res, res, device=self.device)

            for t in range(num_timesteps - 1, -1, -1):
                # get schedule parameters for this timestep
                alpha_bar_t = self.alpha_bars[t]
                alpha_bar_t_prev = self.alpha_bars[t-1] if t > 0 else torch.tensor(1.0, device=self.device)
                alpha_t = self.alphas[t]
                beta_t = self.betas[t]

                # DiT expects a batched timestep tensor, not a single scalar
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=self.device)
                eps_theta = self.model(xt, t_tensor)
                x0_hat = (1 / torch.sqrt(alpha_bar_t)) * (xt - torch.sqrt(1 - alpha_bar_t) * eps_theta)

                mean = (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t) * x0_hat \
                     + (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t) * xt

                # sample noise z
                if t > 0:
                    z  = torch.randn_like(xt)
                    xt = mean + torch.sqrt(beta_t) * z
                else:
                    xt = mean

            # [DiT-VAE] decode denoised latents back to pixel images: (B, 4, 8, 8) -> (B, 3, 64, 64)
            pixel_imgs = decode(xt)

        return pixel_imgs


    # Helper function to prepare the images for the FID
    def _prepare_for_fid(self, imgs_m1_1):
        # [DiT-VAE]no channel repeat: decoded VAE output is already RGB (3 channels)
        #[DiT-VAE] input is 64x64, still resize to 75x75 for InceptionV3
        imgs = (imgs_m1_1 / 2 + 0.5).clamp(0, 1)
        imgs = F.resize(imgs, [75, 75], antialias=True)
        return imgs

    # Helper function to precompute the FID of the real data
    def _precompute_fid_real(self):
        print(
            f"[FID] Pre-computing real features on up to {self.num_fid_samples} samples...")
        total = 0
        # [DiT-VAE] use training_dataloader (batched) instead of dataset (unbatched single items)
        # fid.update() requires (B, C, H, W) shape; iterating dataset directly gives (C, H, W)
        for imgs, _ in self.training_dataloader:
            imgs = imgs.to(self.device)
            imgs = self._prepare_for_fid(imgs)
            self.fid.update(imgs, real=True)
            total += imgs.size(0)
            if total >= self.num_fid_samples:
                break
        print(f"[FID] Real features ready ({total} samples).")

    # Helper function to compute the FID of the generated images
    @torch.no_grad()
    def compute_fid(self):
        """Generate num_fid_samples fake images, return FID float."""
        if self.fid is None:
            return None
        self.model.eval()
        batch = 200
        needed = self.num_fid_samples
        done = 0

        while done < needed:
            n = min(batch, needed - done)
            fake = self.run_inference(num_samples=n)
            fake = self._prepare_for_fid(fake.to(self.device))
            self.fid.update(fake, real=False)
            done += n

        # Move to CPU for compute() since torch.linalg.eigvals requires MAGMA on GPU
        self.fid.cpu()
        val = float(self.fid.compute().item())
        self.fid.to(self.device)

        self.fid.reset()
        self._precompute_fid_real()
        self.model.train()
        return val

# Entry point, auto-resume from checkpoint if one exists ===================================================================================
if __name__ == '__main__':
    ld = LatentDiffusion()
    checkpoint_path = 'checkpoints/checkpoint_VAE.pt'
    if os.path.exists(checkpoint_path):
        print(f'Found checkpoint at {checkpoint_path}, resuming...')
        start_epoch, total_losses, all_fids, fid_epochs = ld.load_checkpoint(checkpoint_path)
        ld.train(start_epoch=start_epoch, total_losses=total_losses,
                 all_fids=all_fids, fid_epochs=fid_epochs)
    else:
        print('No checkpoint found, starting from scratch.')
        ld.train()
