import numpy as np
from tqdm import tqdm
import torch
import math
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
from torch.utils.data import Dataset


### UNET Architecture
class Attention(nn.Module):

    def __init__(self, in_channels, heads = 1, norm_groups = 32):
        super(Attention, self).__init__()

        self.heads = heads

        self.normalization = nn.GroupNorm(norm_groups, in_channels)

        self.q_layer = nn.Linear(in_channels, in_channels)
        self.k_layer = nn.Linear(in_channels, in_channels)
        self.v_layer = nn.Linear(in_channels, in_channels)

        self.softmax = nn.Softmax(dim = -1)

        self.output = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        init_input = x.shape ## [batch_size, in_channels, height, width]
        (batch, channels, h, w) = init_input
        img_size = h * w

        x = x.view((batch, channels, img_size))

        ## Group Normalization
        x_norm = self.normalization(x)
        x_norm = x_norm.permute(0, 2, 1).reshape((-1, channels))

        ## Query, Key, Value.
        qkv_size = (batch, channels, img_size)
        q = self.q_layer(x_norm).view(qkv_size)
        k = self.k_layer(x_norm).view(qkv_size)
        v = self.v_layer(x_norm).view(qkv_size)

        ## Scaled Dot product
        attention = torch.transpose(torch.matmul(q, torch.transpose(k, 1, 2)), 1, 2)

        # Attention scores and weights
        d_k = 1 / np.sqrt(channels)
        scores = d_k * torch.matmul(q, k.transpose(-2, -1))
        weights = self.softmax(scores)

        # Apply attention weights to values and reshape for the Linear layer
        attention = torch.matmul(weights, v).permute(0, 2, 1).reshape((-1, channels))

        output = self.output(attention).view(qkv_size)

        return (x + output).view(init_input)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

## "Attention is All You Need"
## https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
class TimeEmbeddings(nn.Module):
    def __init__(self, in_channels, out_channels = None):
        super(TimeEmbeddings, self).__init__()

        self.half_channels = in_channels // 2

        if out_channels:
            self.only_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                Swish()
            )
        else:
            self.only_mlp = False
            out_channels = in_channels * 4
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                Swish(),
                nn.Linear(out_channels, out_channels)
            )

    def forward(self, x):
        if not self.only_mlp:
            factor = np.log(10000) / (self.half_channels - 1)

            embeddings = torch.exp(torch.arange(self.half_channels) * -factor).to(x.device)
            embeddings = x[:, None] * embeddings[None, :]

            sin = torch.sin(embeddings)
            cos = torch.cos(embeddings)

            embeddings = torch.cat((sin, cos), dim = 1)

        else:
            embeddings = torch.clone(x)

        return self.mlp(embeddings)

class ConvNextBlock(nn.Module):
    def __init__(self, dim_in, dim_out, ad_group_norm = False, extra = 2):
        super().__init__()

        self.time_emb = TimeEmbeddings(256, dim_in)

        exp_dim = dim_in * extra
        
        self.norm1 = nn.LayerNorm(dim_in)
        self.act = nn.GELU()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size = 7, stride = 1, padding = 3, groups = dim_in)
        self.conv2 = nn.Conv2d(dim_in, exp_dim, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(exp_dim, dim_out, kernel_size = 1, stride = 1)
        
        if dim_in != dim_out:
            self.residual = nn.Conv2d(dim_in, dim_out, kernel_size = 1)
        else:
            self.residual = nn.Identity()


    def forward(self, inputs, t):
        ## Depthwise Convolution
        x = self.conv1(inputs)

        ## Time Embeddings
        emb = self.time_emb(t)
        dim1, dim2 = emb.shape
        x += emb.view(dim1, dim2, 1, 1)

        # Normalization
        x = x.permute(0,2,3,1)
        x = self.norm1(x)
        x = x.permute(0,3,1,2)
        
        ## Conv Block with Activation
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        
        return x + self.residual(inputs)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out, ad_group_norm = False):
        super().__init__()
        
        
        self.ad_group_norm = ad_group_norm
        
        self.time_emb = TimeEmbeddings(256, dim_out)
        
        if ad_group_norm:
            self.time_emb = TimeEmbeddings(256, 2 * dim_out)

        self.gn1 = nn.GroupNorm(32, dim_in)
        self.gn2 = nn.GroupNorm(32, dim_out)

        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size = 3, padding = 1)
        self.swish = Swish()

        if dim_in != dim_out:
            self.shortcut = nn.Conv2d(dim_in, dim_out, kernel_size = 1)
        else:
            self.shortcut = nn.Identity()


    def forward(self, inputs, t):
        ## First Conv Block
        x = self.gn1(inputs)
        x = self.swish(x)
        x = self.conv1(x)

        ## Time Embeddings
        emb = self.time_emb(t)
        dim1, dim2 = emb.shape

        ## Adaptive group normalizaiton, as in 'Diffusion Models Beat GANs on Image Synthesis' (2021)
        if self.ad_group_norm:
            scale, shift = torch.chunk(emb.view(dim1, dim2, 1, 1), 2, dim=1)
            x = self.gn2(x) * (1 + scale) + shift
        else:
            x += emb.view(dim1, dim2, 1, 1)
            x = self.gn2(x)

        x = self.swish(x)
        x = self.conv2(x)

        return x + self.shortcut(inputs)

class DownSampleBlock(nn.Module):

    def __init__(self, dim_in, dim_out, block, use_attention = False, downsampling = False, agn = False):
        super().__init__()

        self.use_attention = use_attention
        self.downsampling = downsampling

        self.res1 = block(dim_in, dim_out, ad_group_norm = agn)
        self.res2 = block(dim_out, dim_out, ad_group_norm = agn)


        if self.use_attention:
            self.attention1 = Attention(dim_out)
            self.attention2 = Attention(dim_out)

        if self.downsampling:
            self.downsample = nn.Conv2d(dim_out, dim_out, kernel_size = 3, stride = 2, padding = 1)


    def forward(self, inputs, timestep):
        block1 = self.res1(inputs, timestep)
        if self.use_attention:
            block1 = self.attention1(block1)

        block2 = self.res2(block1, timestep)
        if self.use_attention:
            block2 = self.attention2(block2)


        if self.downsampling:
            x = self.downsample(block2)
            return (block1, block2, x)

        return (block1, block2)

class UpSampleBlock(nn.Module):

    def __init__(self, dim_in, dim_out, img_size, block, use_attention = False, upsampling = False, agn = False):
        super().__init__()

        self.img_size = img_size
        self.use_attention = use_attention
        self.use_upsampling = upsampling

        self.res1 = block(dim_in + dim_in, dim_in, ad_group_norm = agn)
        self.res2 = block(dim_in + dim_in, dim_in, ad_group_norm = agn)
        self.res3 = block(dim_in + dim_out, dim_out, ad_group_norm = agn)

        if self.use_attention:
            self.attention1 = Attention(dim_in)
            self.attention2 = Attention(dim_in)
            self.attention3 = Attention(dim_out)

        if self.use_upsampling:
            self.upsample = nn.ConvTranspose2d(dim_out, dim_out, kernel_size = 4, stride = 2, padding = 1)

    def forward(self, inputs, skips, timestep):
        x = torch.cat([inputs, skips[0]], axis = 1) ## Skip Connection 1

        x = self.res1(x, timestep)
        if self.use_attention:
            x = self.attention1(x)

        x = torch.cat([x, skips[1]], axis = 1) ## Skip Connection 2

        x = self.res2(x, timestep)
        if self.use_attention:
            x = self.attention2(x)


        x = torch.cat([x, skips[2]], axis = 1) ## Skip Connection 3
        x = self.res3(x, timestep)
        if self.use_attention:
            x = self.attention3(x)

        if self.use_upsampling:
            x = self.upsample(x)

        scale_factor = self.img_size / x.shape[-1] #last dim is downscaled shape
        feature_map = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return x, feature_map

class MiddleBlock(nn.Module):

    def __init__(self, dim_in, block, agn = False):
        super().__init__()
        self.res1 = block(dim_in, dim_in, ad_group_norm = agn)
        self.res2 = block(dim_in, dim_in, ad_group_norm = agn)

        self.attention = Attention(dim_in)

    def forward(self, inputs, timestep):
        x = self.res1(inputs, timestep)
        x = self.attention(x)
        x = self.res2(x, timestep)
        return x

class UNet(nn.Module):
    def __init__(self, channels, img_size, block = "res", agn = False, use_attention=True):
        super().__init__()
        
        b = None
        if block == "res":
            b = ResidualBlock
        elif block == "convNext": 
            b = ConvNextBlock
        else:
            raise Exception(f"Block has to be one of the following: `res`, `convNext`")

        self.img_size = img_size
        self.img_channels = channels
        
        self.image_projection = nn.Conv2d(channels, 64, kernel_size = 3, padding = 1)
        self.time_emb = TimeEmbeddings(64)

        self.use_attention = use_attention


        # Downsample
        self.down_blocks = nn.ModuleList([
            DownSampleBlock(64, 64, downsampling = True, agn = agn, block = b),
            DownSampleBlock(64, 128, downsampling = True, agn = agn, block = b),
            DownSampleBlock(128, 256, downsampling = True, agn = agn, block = b),
            DownSampleBlock(256, 1024, use_attention = True, agn = agn, block = b)
        ])


        self.middle_block = MiddleBlock(1024, agn = agn, block = b)

        self.up_blocks = nn.ModuleList([
            UpSampleBlock(1024, 256, use_attention = self.use_attention, upsampling = True, img_size = img_size, agn = agn, block = b),
            UpSampleBlock(256, 128, upsampling = True, img_size = img_size, agn = agn, block = b),
            UpSampleBlock(128, 64, upsampling = True, img_size = img_size, agn = agn, block = b),
            UpSampleBlock(64, 64, img_size = img_size, agn = agn, block = b)
        ])

        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            Swish(),
            nn.Conv2d(64, channels, kernel_size = 3, padding = 1)
        )

    def forward(self, inputs, timestep):

        x = self.image_projection(inputs)
        t = self.time_emb(timestep)

        ### [img_proj, Block1_Res1, Block1_Res2, Block1_Downsampling, Block2_Res1 ... Block4_Res2]
        down_outputs = [x]


        for block_idx, down_layer in enumerate(self.down_blocks):
            outputs = down_layer(x, t)

            for output in outputs:
                down_outputs.insert(0, output)

            x = outputs[-1]

        x = self.middle_block(x, t)

        feature_maps = []

        for block_idx, up_layer in enumerate(self.up_blocks):
            skips = [down_outputs.pop(0) for _ in range(3)]
            
            x, feature_map = up_layer(x, skips, t)
            feature_maps.append(feature_map)


        concat_f_maps = torch.cat(feature_maps, dim=1)
        return self.output(x), concat_f_maps

### Adopted from: https://nn.labml.ai/diffusion/ddpm/index.html
def gather(consts, t):
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DDPM:
    def __init__(self, model, T, cosine_schedule = False, loss = 'mse', device = 'cpu'):
        self.model = model
        self.device = device

        self.T = T
        
        if cosine_schedule:
            self.beta = self.create_cosine_beta_schedule(T)
        else:
            self.beta = torch.linspace(0.0001, 0.02, self.T).to(device)
        
        self.beta = torch.linspace(0.0001, 0.02, self.T).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma2 = self.beta

        if loss == 'mse':
            self.loss_fn = F.mse_loss
        else:
            raise Exception("Wrong Loss Function")

    def compute_mean_var(self, x0, t):
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1 - gather(self.alpha_bar, t)
        return mean, var

    def approx_post_sample(self, x0, t, eps):
        mean, var = self.compute_mean_var(x0, t)
        return mean + torch.sqrt(var) * eps

    def sample_timestep(self, xt, timestep):
        output, _ = self.model(xt, timestep)

        alpha_bar = gather(self.alpha_bar, timestep)
        alpha = gather(self.alpha, timestep)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * output)
        var = gather(self.sigma2, timestep)

        eps = torch.randn(xt.shape, device=xt.device)
        return mean + torch.sqrt(var) * eps

    @torch.no_grad()
    def sample(self, n_samples = 6):

        print("Sampling...")
        
        img_size = self.model.img_size
        img_channels = self.model.img_channels
        
        x = torch.randn([n_samples, img_channels, img_size, img_size]).to(self.device)

        # Remove noise for $T$ steps
        for t_i in tqdm(range(self.T)):
            t = self.T - t_i - 1
            x = self.sample_timestep(x, x.new_full((n_samples,), t, dtype=torch.long))

        return x

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device, dtype=torch.long)

        random_noise = torch.randn_like(x0)

        xt = self.approx_post_sample(x0, t, eps = random_noise)
        output, _ = self.model(xt, t)

        return self.loss_fn(random_noise, output)

    
    def create_cosine_beta_schedule(self, T, max_beta=0.999, s=0.008):
        """
        Create an improved beta schedule as in 'Improved Denoising Diffusion models' (2021)
        """
        pi = torch.Tensor([math.pi]).to(self.device)
        alpha_line = lambda t: torch.cos((t + s) / (1 + s) * pi / 2)**2
        
        betas = []
        for t in range(self.T):
            t1 = t / T
            t2 = (t+1) / T
            betas.append(min(1 - alpha_line(t2) / alpha_line(t1), max_beta))
    
        return torch.Tensor(betas).to(self.device)

def plot_samples(samples):
    fig, axes = plt.subplots(nrows=1, ncols=samples.shape[0])

    for i, img in enumerate(samples):
        img_to_show = img.permute(1,2,0).detach().cpu()
        img_to_show = torch.clamp(img_to_show, -1.0, 1.0)

        axes[i].imshow(img_to_show)

    plt.show()