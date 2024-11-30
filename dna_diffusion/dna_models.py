from functools import partial

from memory_efficient_attention_pytorch import Attention as EfficientAttention
import torch
from torch import nn
import torch.nn.functional as F
import os 
import sys
import numpy as np

file_dir = os.path.dirname(__file__)
dna_diff_path = os.path.join(file_dir, '..', '..','re_design', 'DNA-Diffusion', 'src')
if dna_diff_path not in sys.path:
    sys.path.append(dna_diff_path)

from dnadiffusion.utils.utils import default, extract, linear_beta_schedule
from dnadiffusion.models.layers import (
    ResnetBlock, 
    LearnedSinusoidalPosEmb, 
    Residual, 
    PreNorm, 
    LinearAttention, 
    Downsample, 
    Attention, 
    Upsample)


class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        timesteps,
        masking=True,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.masking = masking

        # Diffusion params
        betas = linear_beta_schedule(timesteps, beta_end=0.2)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @property
    def device(self):
        return self.betas.device

    @torch.no_grad()
    def sample(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
        )

    @torch.no_grad()
    def sample_cross(self, classes, shape, cond_weight):
        return self.p_sample_loop(
            classes=classes,
            image_size=shape,
            cond_weight=cond_weight,
            get_cross_map=True,
        )

    @torch.no_grad()
    def p_sample_loop(self, classes, image_size, cond_weight, get_cross_map=False, get_clean=True):
        b = image_size[0]
        device = self.device

        img = torch.randn(image_size, device=device)
        imgs = []
        cross_images_final = []

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            # classes = classes.repeat(2)
            # context_mask = context_mask.repeat(2)
            classes = torch.repeat_interleave(classes, repeats=2, dim=0)
            context_mask = torch.repeat_interleave(context_mask, repeats=2, dim=0)
            if self.masking:
                context_mask[n_sample:] = 0.0
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )

        else:
            sampling_fn = partial(self.p_sample)

        for i in reversed(range(0, self.timesteps)):
            img, cross_matrix = sampling_fn(x=img, t=torch.full((b,), i, device=device, dtype=torch.long), t_index=i)
            imgs.append(img.cpu().numpy())
            cross_images_final.append(cross_matrix.cpu().numpy())

        if get_cross_map:
            return imgs, cross_images_final
        else:
            return imgs

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, time=t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided(self, x, classes, t, t_index, context_mask, cond_weight):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        batch_size = x.shape[0]
        device = self.device
        # double to do guidance with
        t_double = t.repeat(2).to(device)
        x_double = x.repeat(2, 1, 1, 1).to(device)
        betas_t = extract(self.betas, t_double, x_double.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape, device)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape, device)

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
    
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.float)
        # model = self.accelerator.unwrap_model(self.model)
        self.model.output_attention = True
        preds, cross_map_full = self.model(x_double, time=t_double, classes=classes_masked)
        self.model.output_attention = False
        cross_map = cross_map_full[:batch_size]
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean, cross_map
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape, device)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise, cross_map

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, torch.randn_like(x_start))
        device = self.device

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape, device)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape, device)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, classes, noise=None, loss_type="huber", p_uncond=0.1):
        device = self.device
        noise = default(noise, torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.masking: # Mask for unconditional guidance
            context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - p_uncond)).to(device)
            classes = classes * context_mask[:, np.newaxis]
            # classes = classes.type(torch.float)
        
        predicted_noise = self.model(x_noisy, t, classes)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, classes):
        device = self.device
        # x = x.type(torch.float32)
        # classes = classes.type(torch.float)
        b = x.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=device).long()

        return self.p_losses(x, t, classes)



class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        dim_mults: tuple = (1, 2, 4),
        channels: int = 1,
        resnet_block_groups: int = 8,
        label_vector_size:int = 51,
        # label_embed_size:int = 1000,
        learned_sinusoidal_dim: int = 18,
        # num_classes: int = 10,
        output_attention: bool = False,
    ) -> None:
        super().__init__()

        # determine dimensions
        self.seq_length = dim
        self.channels = channels
        # if you want to do self conditioning uncomment this
        input_channels = channels
        self.output_attention = output_attention

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, (7, 7), padding=3)
        dims = [init_dim, *(dim * m for m in dim_mults)]

        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        ### MY CODE
        self.label_emb = nn.Sequential(
            nn.Linear(label_vector_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        # self.label_emb = nn.Embedding(label_embed_size, time_dim)

        # if num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_dim)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.cross_attn = EfficientAttention(
            dim=self.seq_length,
            dim_head=64,
            heads=1,
            memory_efficient=True,
            q_bucket_size=1024,
            k_bucket_size=2048,
        )
        self.norm_to_cross = nn.LayerNorm(dim * 4)

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):
        x = self.init_conv(x)
        r = x.clone()

        t_start = self.time_mlp(time)
        t_mid = t_start.clone()
        t_end = t_start.clone()
        t_cross = t_start.clone()

        if classes is not None:
            t_start += self.label_emb(classes)
            t_mid += self.label_emb(classes)
            t_end += self.label_emb(classes)
            t_cross += self.label_emb(classes)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_start)
            h.append(x)

            x = block2(x, t_start)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_mid)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_mid)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_mid)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t_end)
        x = self.final_conv(x)
        x_reshaped = x.reshape(-1, 4, self.seq_length)
        t_cross_reshaped = t_cross.reshape(-1, 4, self.seq_length)

        crossattention_out = self.cross_attn(
            self.norm_to_cross(x_reshaped.reshape(-1, 4 * self.seq_length)).reshape(-1, 4, self.seq_length),
            context=t_cross_reshaped,
        )  # (-1,1, 4, seq_length)
        crossattention_out = crossattention_out.view(-1, 1, 4, self.seq_length)
        
        try:
            x = x + crossattention_out
        except BaseException as e:
            print("-"*20)
            print(x.shape)
            print(crossattention_out.shape)
            raise e

        if self.output_attention:
            return x, crossattention_out
        return x
