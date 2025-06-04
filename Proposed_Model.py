# deep_steganography.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
import math
from skimage.metrics import structural_similarity as ssim_np
import lpips

# ==== CONFIGURATION ====
# Paths
MODELS_PATH = 'models/Exp1_models'
TRAIN_PATH = 'Dataset/train'
TEST_PATH = 'Dataset/test'
OUTPUT_IMAGE_PATH = 'output/Exp1/'

# Hyperparameters
num_epochs = 120 #20
batch_size = 2 #8
learning_rate = 0.000005 #0.0001
beta = 5 #1

# ImageNet normalization stats
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Ensure output directories exist =======
#os.makedirs(MODELS_PATH, exist_ok=True)
#os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)

# ======= Utility Functions =======
def gaussian(tensor, mean=0, stddev=0.1):
    noise = torch.nn.init.normal_(torch.empty_like(tensor), mean, stddev)
    return Variable(tensor + noise)

def denormalize(image, std, mean):
    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    image = torch.clamp(image, 0, 1)
    return image

def imshow(img, idx, learning_rate, beta):
    img = denormalize(img.cpu(), std, mean)
    npimg = img.numpy()
    if npimg.shape[0]==3:
        npimg=(np.transpose(npimg, (1, 2, 0)))
        
        
    plt.imshow(npimg)
    plt.title(f'Example {idx}, lr={learning_rate}, B={beta}')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_IMAGE_PATH, f'example_{idx}.png'))
    plt.close()
    
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()
def compute_metrics(img1, img2):
    """
    img1, img2: tensors of shape (1, 3, H, W), values in [0,1]
    Returns: dict with mse, psnr, ssim, lpips
    """
    
    if img1.shape[2:] != img2.shape[2:]:
        img2 = F.interpolate(img2, size=img1.shape[2:], mode='bilinear', align_corners=False)
    
    # MSE
    mse_val = torch.mean((img1 - img2) ** 2).item()

    # PSNR
    psnr_val = 20 * math.log10(1.0) - 10 * math.log10(mse_val) if mse_val != 0 else float('inf')

    # SSIM (convert to numpy)
    img1_np = img1.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    ssim_val = ssim_np(img1_np, img2_np, channel_axis=2, data_range=1.0)

    # LPIPS requires inputs in [-1, 1]
    img1_lp = (img1 * 2 - 1).to(img1.device)
    img2_lp = (img2 * 2 - 1).to(img2.device)
    lpips_val = lpips_model(img1_lp, img2_lp).item()

    return {
        "mse": mse_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": lpips_val
    }

# ======= DATA LOADING =======
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TRAIN_PATH, transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TEST_PATH, transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

class ResidualBlockAfterTwoConvs(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(ResidualBlockAfterTwoConvs, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        if out.shape != identity.shape:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + identity
       

# ======= Define the model (import or define networks here) =======
# ... (Networks defined previously: PrepNetwork, HidingNetwork, RevealNetwork, Net)
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalP3 = nn.Sequential(nn.Conv2d(150, 50, kernel_size=3, padding=1), nn.ReLU())
        self.finalP4 = nn.Sequential(nn.Conv2d(150, 50, kernel_size=4, padding=1), nn.ReLU(),
                                    nn.Conv2d(50, 50, kernel_size=4, padding=2), nn.ReLU())
        self.finalP5 = nn.Sequential(nn.Conv2d(150, 50, kernel_size=5, padding=2), nn.ReLU())

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()

        self.initialH3 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1)
        )

        self.initialH4 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1)
        )

        self.initialH5 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2)
        )

        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU()
        )
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0)
        )

    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        
        # 🔧 Fix mismatch before concatenation
        min_h = min(h1.shape[2], h2.shape[2], h3.shape[2])
        min_w = min(h1.shape[3], h2.shape[3], h3.shape[3])

        h1 = F.interpolate(h1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        h2 = F.interpolate(h2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        h3 = F.interpolate(h3, size=(min_h, min_w), mode='bilinear', align_corners=False)



        mid = torch.cat((h1, h2, h3), dim=1)

        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)

        mid2 = torch.cat((h4, h5, h6), dim=1)
        out = self.finalH(mid2)

        out_noise = out
        return out, out_noise

class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()

        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1)
        )

        self.initialR4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1)
        )

        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2)
        )

        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU()
        )
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0)
        )

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
	# 🔧 Fix mismatch before concatenation
        min_h = min(r1.shape[2], r2.shape[2], r3.shape[2])
        min_w = min(r1.shape[3], r2.shape[3], r3.shape[3])

        r1 = F.interpolate(r1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        r2 = F.interpolate(r2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        r3 = F.interpolate(r3, size=(min_h, min_w), mode='bilinear', align_corners=False)

        mid = torch.cat((r1, r2, r3), dim=1)

        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)

        mid2 = torch.cat((r4, r5, r6), dim=1)
        out = self.finalR(mid2)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2, x_2_noise = self.m2(mid)
        x_3 = self.m3(x_2_noise)
        return x_2, x_3

# Instantiate model
net = Net().to(device)

# ======= LOSS FUNCTION =======
def customized_loss(S_prime, C_prime, S, C, B):
    #c_prime=train_output
    #c=train_covers
    #S_prime = train_hidden
    #S = train_secrets
    if C.shape[2:] != C_prime.shape[2:]:
        C = F.interpolate(C, size=C_prime.shape[2:], mode='bilinear', align_corners=False)
    if S.shape[2:] != S_prime.shape[2:]:
        S = F.interpolate(S, size=S_prime.shape[2:], mode='bilinear', align_corners=False)
    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret

# ======= TRAINING LOOP =======
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_history = []

print("Starting training...")

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for idx, (data, _) in enumerate(train_loader):
        # Split batch into secret and cover images
        train_secrets = data[:len(data)//2].to(device)
        train_covers = data[len(data)//2:].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        train_hidden, train_output = net(train_secrets, train_covers)

        # Loss computation
        total_loss, loss_cover, loss_secret = customized_loss(train_output, train_hidden, train_secrets, train_covers, beta)

        # Backward pass & optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        loss_history.append(total_loss.item())

        # Print batch progress
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{idx}/{len(train_loader)}]: Total Loss: {total_loss.item():.4f}")

    # Save model after each epoch
    torch.save(net.state_dict(), os.path.join(MODELS_PATH, f'epoch_{epoch+1}.pth'))
    print(f"Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f}")

print("Training completed!\n")

# Initialize LPIPS model (AlexNet backbone)
lpips_model = lpips.LPIPS(net='alex').to(device)
lpips_model.eval()

# ======= TESTING & IMAGE SAVING =======
print("Starting testing & saving output images...")
net.eval()
test_losses = []
all_metrics = []  # For storing MSE, PSNR, SSIM, LPIPS for each image

with torch.no_grad():
    for idx, (data, _) in enumerate(test_loader):
        test_secret = data[:len(data)//2].to(device)
        test_cover = data[len(data)//2:].to(device)
    
        # Forward pass
        test_hidden, test_output = net(test_secret, test_cover)
        
        # Clamp outputs to [0,1] for valid metrics
        test_output = torch.clamp(test_output, 0, 1)
        test_hidden = torch.clamp(test_hidden, 0, 1)
        
        # Loss
        test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)
        test_losses.append(test_loss.item())
        
        # Compute metrics (recovered vs secret image)
        metrics = compute_metrics(test_output[0].unsqueeze(0), test_secret[0].unsqueeze(0))
        all_metrics.append(metrics)

        print(f"[{idx}] MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")
        
        # Save images for visualization
        save_image(test_secret.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'secret_{idx}.png'))
        save_image(test_output.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'revealed_{idx}.png'))
        save_image(test_cover.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'cover_{idx}.png'))
        save_image(test_hidden.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'stego_{idx}.png'))

        # Visual display image
        #imgs = [test_secret.cpu(), test_output.cpu(), test_cover.cpu(), test_hidden.cpu()]
        #imgs_tsor = torch.cat(imgs, 0)
        #imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)
        imgs = [test_secret.cpu(), test_output.cpu(), test_cover.cpu(), test_hidden.cpu()]

        # 🔧 Fix: Resize all images to same H×W before concatenation
        min_h = min(img.shape[2] for img in imgs)
        min_w = min(img.shape[3] for img in imgs)

        imgs_resized = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)  # [1, C, H, W]
            resized = F.interpolate(img, size=(min_h, min_w), mode='bilinear', align_corners=False)
            imgs_resized.append(resized.squeeze(0))  # back to [C, H, W]


        imgs_tsor = utils.make_grid(imgs_resized, nrow=4)
        imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)

        print(f"Test Batch [{idx}/{len(test_loader)}]: Total Loss: {test_loss.item():.4f}")

mean_test_loss = np.mean(test_losses)
print(f"Average Test Loss: {mean_test_loss:.4f}")

# Average of the 4 metrics
avg_metrics = {
    "mse": np.mean([m["mse"] for m in all_metrics]),
    "psnr": np.mean([m["psnr"] for m in all_metrics]),
    "ssim": np.mean([m["ssim"] for m in all_metrics]),
    "lpips": np.mean([m["lpips"] for m in all_metrics])
}
print("\n=== Average Metrics Over Test Set ===")
print(f"MSE   : {avg_metrics['mse']:.6f}")
print(f"PSNR  : {avg_metrics['psnr']:.2f} dB")
print(f"SSIM  : {avg_metrics['ssim']:.4f}")
print(f"LPIPS : {avg_metrics['lpips']:.4f}")

# ======= OPTIONAL: Save loss history for analysis =======
np.save(os.path.join(OUTPUT_IMAGE_PATH, 'loss_history.npy'), np.array(loss_history))
print("All done! Output images and models saved.")

'''



# deep_steganography.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
import lpips
from skimage.metrics import structural_similarity as ssim_np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image
import math
from skimage.metrics import structural_similarity as ssim_np
import lpips

# ==== CONFIGURATION ====
# Paths
MODELS_PATH = 'models/test5'
TRAIN_PATH = 'Dataset/train'
TEST_PATH = 'Dataset/test'
OUTPUT_IMAGE_PATH = 'output/test5/'

# Hyperparameters
num_epochs = 70 #20
batch_size = 2 #8
learning_rate = 0.0001 #0.0001
beta = 1 #1
alpha=0.5

# ImageNet normalization stats
std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Ensure output directories exist =======
#os.makedirs(MODELS_PATH, exist_ok=True)
#os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)

# ======= Utility Functions =======
def gaussian(tensor, mean=0, stddev=0.1):
    noise = torch.nn.init.normal_(torch.empty_like(tensor), mean, stddev)
    return Variable(tensor + noise)

def denormalize(image, std, mean):
    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    image = torch.clamp(image, 0, 1)
    return image

def imshow(img, idx, learning_rate, beta):
    img = denormalize(img.cpu(), std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Example {idx}, lr={learning_rate}, B={beta}')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_IMAGE_PATH, f'example_{idx}.png'))
    plt.close()

# ======= DATA LOADING =======
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TRAIN_PATH, transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(TEST_PATH, transform),
    batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
)

class ResidualBlockAfterTwoConvs(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super(ResidualBlockAfterTwoConvs, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        if out.shape != identity.shape:
            identity = F.interpolate(identity, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + identity
       

# ======= Define the model (import or define networks here) =======
# ... (Networks defined previously: PrepNetwork, HidingNetwork, RevealNetwork, Net)
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU()
        )
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU()
        )
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.GELU()
        )

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out

class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()

        self.initialH3 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1)
        )

        self.initialH4 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1)
        )

        self.initialH5 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2)
        )

        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU()
        )
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0)
        )

    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)

        min_h = min(h1.shape[2], h2.shape[2], h3.shape[2])
        min_w = min(h1.shape[3], h2.shape[3], h3.shape[3])

        h1 = F.interpolate(h1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        h2 = F.interpolate(h2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        h3 = F.interpolate(h3, size=(min_h, min_w), mode='bilinear', align_corners=False)

        mid = torch.cat((h1, h2, h3), dim=1)

        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)

        mid2 = torch.cat((h4, h5, h6), dim=1)
        out = self.finalH(mid2)

        out_noise = out
        return out, out_noise

class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()

        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=3, padding=1)
        )

        self.initialR4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=4, padding=1)
        )

        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.GELU(),
            ResidualBlockAfterTwoConvs(50, kernel_size=5, padding=2)
        )

        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.GELU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.GELU()
        )
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.GELU()
        )
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0)
        )

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)

        min_h = min(r1.shape[2], r2.shape[2], r3.shape[2])
        min_w = min(r1.shape[3], r2.shape[3], r3.shape[3])

        r1 = F.interpolate(r1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        r2 = F.interpolate(r2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        r3 = F.interpolate(r3, size=(min_h, min_w), mode='bilinear', align_corners=False)

        mid = torch.cat((r1, r2, r3), dim=1)

        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)

        mid2 = torch.cat((r4, r5, r6), dim=1)
        out = self.finalR(mid2)

        return out



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2, x_2_noise = self.m2(mid)
        x_3 = self.m3(x_2_noise)
        return x_2, x_3

# Instantiate model
net = Net().to(device)

from pytorch_msssim import ssim
import torch.nn.functional as F

def customized_loss(S_prime, C_prime, S, C, beta, alpha=0.8):
    """
    Computes combined MSE + SSIM loss for cover and secret, after resizing all tensors to match spatial shape.

    Parameters:
        S_prime: Revealed secret image
        C_prime: Stego image
        S: Original secret image
        C: Original cover image
        beta: Weight for secret loss
        alpha: MSE vs SSIM weighting

    Returns:
        loss_all: Total loss
        loss_cover: Cover image loss
        loss_secret: Secret image loss
    """
    # 🔧 Resize all to smallest shared spatial shape
    min_h = min(C.shape[2], C_prime.shape[2], S.shape[2], S_prime.shape[2])
    min_w = min(C.shape[3], C_prime.shape[3], S.shape[3], S_prime.shape[3])

    C = F.interpolate(C, size=(min_h, min_w), mode='bilinear', align_corners=False)
    C_prime = F.interpolate(C_prime, size=(min_h, min_w), mode='bilinear', align_corners=False)
    S = F.interpolate(S, size=(min_h, min_w), mode='bilinear', align_corners=False)
    S_prime = F.interpolate(S_prime, size=(min_h, min_w), mode='bilinear', align_corners=False)

    # Cover loss: Stego vs Cover
    mse_cover = F.mse_loss(C_prime, C)
    ssim_cover = 1 - ssim(C_prime, C, data_range=1.0)
    loss_cover = alpha * mse_cover + (1 - alpha) * ssim_cover

    # Secret loss: Revealed vs Secret
    mse_secret = F.mse_loss(S_prime, S)
    ssim_secret = 1 - ssim(S_prime, S, data_range=1.0)
    loss_secret = alpha * mse_secret + (1 - alpha) * ssim_secret

    # Total loss
    loss_all = loss_cover + beta * loss_secret
    return loss_all, loss_cover, loss_secret

# ======= TRAINING LOOP =======
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
loss_history = []

print("Starting training...")

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0

    for idx, (data, _) in enumerate(train_loader):
        # Split batch into secret and cover images
        train_secrets = data[:len(data)//2].to(device)
        train_covers = data[len(data)//2:].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        train_hidden, train_output = net(train_secrets, train_covers)

        # Loss computation
        total_loss, loss_cover, loss_secret = customized_loss(train_output, train_hidden, train_secrets, train_covers, beta)

        # Backward pass & optimization
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        loss_history.append(total_loss.item())

        # Print batch progress
        if idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{idx}/{len(train_loader)}]: Total Loss: {total_loss.item():.4f}")

    # Save model after each epoch
    torch.save(net.state_dict(), os.path.join(MODELS_PATH, f'epoch_{epoch+1}.pth'))
    print(f"Epoch {epoch+1} completed. Average Loss: {running_loss/len(train_loader):.4f}")

print("Training completed!\n")

# ======= TESTING & IMAGE SAVING =======
print("Starting testing & saving output images...")
net.load_state_dict(torch.load(os.path.join(MODELS_PATH, 'epoch_70.pth')))

net.eval()
test_losses = []

with torch.no_grad():
    for idx, (data, _) in enumerate(test_loader):
        test_secret = data[:len(data)//2].to(device)
        test_cover = data[len(data)//2:].to(device)

        # Forward pass
        test_hidden, test_output = net(test_secret, test_cover)

        # Loss
        test_loss, loss_cover, loss_secret = customized_loss(test_output, test_hidden, test_secret, test_cover, beta)
        test_losses.append(test_loss.item())

        # Save images for visualization
        save_image(test_secret.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'secret_{idx}.png'))
        save_image(test_output.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'revealed_{idx}.png'))
        save_image(test_cover.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'cover_{idx}.png'))
        save_image(test_hidden.cpu(), os.path.join(OUTPUT_IMAGE_PATH, f'stego_{idx}.png'))

        # Visual display image
        #imgs = [test_secret.cpu(), test_output.cpu(), test_cover.cpu(), test_hidden.cpu()]
        imgs = [test_secret.cpu(), test_output.cpu(), test_cover.cpu(), test_hidden.cpu()]

        # 🔧 Resize all to same H×W
        min_h = min(img.shape[2] for img in imgs)
        min_w = min(img.shape[3] for img in imgs)

        imgs_resized = []
        for img in imgs:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            resized = F.interpolate(img, size=(min_h, min_w), mode='bilinear', align_corners=False)
            imgs_resized.append(resized)
        #imgs_tsor = torch.cat(imgs, 0)
        #imshow(utils.make_grid(imgs_tsor), idx+1, learning_rate=learning_rate, beta=beta)
        # Stack along batch dimension
        imgs_batch = torch.cat(imgs_resized, dim=0)  # shape: [B=4, C, H, W]

        # Create grid and show
        imgs_grid = utils.make_grid(imgs_batch, nrow=4)
        imshow(imgs_grid, idx+1, learning_rate=learning_rate, beta=beta)

        print(f"Test Batch [{idx}/{len(test_loader)}]: Total Loss: {test_loss.item():.4f}")

mean_test_loss = np.mean(test_losses)
print(f"Average Test Loss: {mean_test_loss:.4f}")

# ======= OPTIONAL: Save loss history for analysis =======
np.save(os.path.join(OUTPUT_IMAGE_PATH, 'loss_history.npy'), np.array(loss_history))
print("All done! Output images and models saved.")

