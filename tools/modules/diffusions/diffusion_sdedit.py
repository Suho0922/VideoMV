import torch
import math
from PIL import Image
import numpy as np
from utils.registry_class import DIFFUSION
from .schedules import beta_schedule
from .losses import kl_divergence, discretized_gaussian_log_likelihood
from scipy.ndimage import distance_transform_edt
def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if tensor.device != x.device:
        tensor = tensor.to(x.device)
    return tensor[t].view(shape).to(x)


@DIFFUSION.register_class()
class SDEditDDIM(object):
    def __init__(self,
                 schedule='linear_sd',
                 schedule_param={},
                 mean_type='eps',
                 var_type='learned_range',
                 loss_type='mse',
                 epsilon = 1e-12,
                 rescale_timesteps=False,
                 noise_strength=0.0, 
                 device='cuda',
                 **kwargs):
        # check input
        # check input
        assert mean_type in ['x0', 'x_{t-1}', 'eps', 'v']
        assert var_type in ['learned', 'learned_range', 'fixed_large', 'fixed_small']
        assert loss_type in ['mse', 'rescaled_mse', 'kl', 'rescaled_kl', 'l1', 'rescaled_l1','charbonnier']
        
        betas = beta_schedule(schedule, **schedule_param)
        assert min(betas) > 0 and max(betas) <= 1

        if not isinstance(betas, torch.DoubleTensor):
            betas = torch.tensor(betas, dtype=torch.float64)

        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type # eps
        self.var_type = var_type # 'fixed_small'
        self.loss_type = loss_type # mse
        self.epsilon = epsilon # 1e-12
        self.rescale_timesteps = rescale_timesteps # False
        self.noise_strength = noise_strength # 0.0

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])
        
        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        self.timesteps = None
        self.device = device
        self.middle_time_steps = None
    

    def sample_loss(self, x0, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
            if self.noise_strength > 0:
                b, c, f, _, _= x0.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=x0.device)
                noise = noise + self.noise_strength * offset_noise
        return noise



    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0).
        """
        # noise = torch.randn_like(x0) if noise is None else noise
        noise = self.sample_loss(x0, noise)
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + \
               _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0).
        """
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var
    
    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0).
        """
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var
    
    @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = torch.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * torch.exp(0.5 * log_var) * noise
        return xt_1, x0
    
    @torch.no_grad()
    def p_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1).
        """
        # prepare input
        b = noise.size(0)
        xt = noise
        
        # diffusion process
        for step in torch.arange(self.num_timesteps).flip(0):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt
    
    def p_mean_variance(self, xt, t, model, autoencoder=None, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t).
        """
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2

            # import ipdb; ipdb.set_trace()
            
            y_out = model(xt, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, **model_kwargs[0])
            
            u_out = model(xt, self._scale_timesteps(t), autoencoder=autoencoder, sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod, \
            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod, **model_kwargs[1])

            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) 
        
        # compute variance
        if self.var_type == 'learned':
            out, log_var = out.chunk(2, dim=1)
            var = torch.exp(log_var)
        elif self.var_type == 'learned_range':
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(torch.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = torch.exp(log_var)
        elif self.var_type == 'fixed_large':
            var = _i(torch.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = torch.log(var)
        elif self.var_type == 'fixed_small':
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        if autoencoder is not None: 
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        else:
            # compute mean and x0
            if self.mean_type == 'x_{t-1}':
                mu = out  # x_{t-1}
                x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - \
                    _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
            elif self.mean_type == 'x0':
                x0 = out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
            elif self.mean_type == 'eps':
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
            elif self.mean_type == 'v':
                x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out
                mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        
        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1).clamp_(1.0).view(-1, 1, 1, 1)
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    @torch.no_grad()
    def ddim_sample(self, xt, t, model, autoencoder=None, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """

        stride = self.num_timesteps // ddim_timesteps
        
        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, autoencoder, model_kwargs, clamp, percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = torch.randn_like(xt)
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0
    
    @torch.no_grad()
    def ddim_sample_loop(self, noise, model, autoencoder=None, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        for idx, step in enumerate(steps):
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            if idx in [20,30,40]:
                xt, _ = self.ddim_sample(xt, t, model, autoencoder, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
            else:
                xt, _ = self.ddim_sample(xt, t, model, None, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta)
        return xt
    
    
    ################################ added ####################################
    @torch.no_grad()
    def sdedit_sample_loop(self, model, original_image, mask, num_inference_steps, 
                           autoencoder, use_autoencoder=False, model_kwargs={}, clamp=None, percentile=None, 
                           condition_fn=None, guide_scale=None, ddim_timesteps=20, eta=0.0,
                           middle_time_steps=1000, jump_length=1, jump_n_sample=1, add_small_steps=True):
        
        # original_image  [1, C, F, H, W] : [1, 3, 24, 256, 256]
        # prepare input
        b = original_image.size(0) # batch size
        # [1, C, F, H, W] -> [F, C, H, W]
        # import ipdb; ipdb.set_trace()
        original_image = original_image.squeeze(0).permute(1, 0, 2, 3).contiguous()
        latents = autoencoder.encode_firsr_stage(original_image, 0.18215) # [F, C, H, W]
        # [F, C, H, W] -> [1, C, F, H, W]
        original_image = original_image.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        
        # [F,C,H,W] - >[1, C, F, H, W] : [1, 4, 24, 32, 32]
        latents = latents.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
        
        original_latents = latents.clone()
        
        if use_autoencoder:
            autoencoder = autoencoder
        else:
            autoencoder = None
        
        
        
        # xt = noise # [B, C, F, H, W] latent of image
        if middle_time_steps == 1000:
            xt = torch.randn_like(latents) # [B, C, F, H, W] latent of image
        else: 
            xt = self.q_sample(latents, middle_time_steps, noise=None) # [B, C, F, H, W] latent of image
        
        mask = torch.nn.functional.interpolate(mask, size=(xt.shape[2], xt.shape[3], xt.shape[4]), mode='nearest')
        
        mask = self.apply_distance_transform(mask, max_distance=5)
        # set timesteps
        self.set_timesteps(num_inference_steps, middle_time_steps, jump_length, jump_n_sample, add_small_steps=True)
        
        timesteps = self.timesteps
        
        # diffusion process
        for i, t in enumerate(timesteps):
            if  i < len(timesteps) - 1 and timesteps[i + 1] < t:
                prev_timestep = timesteps[i + 1] #if i > 0 else -1
            elif i == len(timesteps) - 1:
                prev_timestep = -1
            elif i < len(timesteps) - 1 and timesteps[i + 1] > t:
                # prev_timestep = None
                prev_timestep = timesteps[i + 1] #if i > 0 else -1
            else:
                raise ValueError("Invalid timesteps same value")
            if prev_timestep :
                if t < prev_timestep:
                    # compute the reverse: x_t-1 -> x_t
                    latents = self.undo_step(latents, t, prev_timestep, generator=None)
                    # t_last = t
                    continue
            
            # import ipdb; ipdb.set_trace()
            t = torch.full((b, ), t, dtype=torch.long, device=xt.device)
            prev_timestep = torch.full((b, ), prev_timestep, dtype=torch.long, device=xt.device)
            
            # 여기에 특정 index에 대해서 autoencoder를 사용할지 말지 결정하는 코드 추가해야함 
            # autoencoder를 사용하는 경우 unet에서 3d_aware_denoising을 함. (lgm으로 3dgs 예측해서 render하고 다시 x_t_1로 보냄)
            latents, _ = self.masked_ddim_step(latents, t, model, original_latents, mask, prev_t=prev_timestep, 
                                            autoencoder=autoencoder, model_kwargs=model_kwargs, clamp=clamp, 
                                            percentile=percentile, condition_fn=condition_fn, guide_scale=guide_scale, 
                                            ddim_timesteps=ddim_timesteps, eta=eta)
        return latents
            
    
    def undo_step(self, xt, t, prev_t, generator=None):
        r"""Sample from q(x_{t+1} | x_t) """
        n = prev_t - t # prev_t > t
        
        for i in range(n):
            beta = self.betas[t + i]
            
            noise = torch.randn_like(xt, dtype=xt.dtype, device=xt.device)
            
            xt = (1 - beta) ** 0.5 * xt + (beta ** 0.5) * noise
        
        return xt
    
    def masked_ddim_step(   
                            self, 
                            xt,
                            t,
                            model,
                            original_image,
                            mask,
                            prev_t=None ,
                            autoencoder=None,
                            model_kwargs={},
                            clamp=None,
                            percentile=None,
                            condition_fn=None,
                            guide_scale=None,
                            ddim_timesteps=20,
                            eta=0.0,
                        ):
        
        if prev_t is None:
            prev_t = self.middle_time_steps // self.num_inference_steps

        _, _, _, x0 = self.p_mean_variance(xt, t, model, autoencoder, model_kwargs, clamp, percentile, guide_scale)
        
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
        
        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (prev_t).clamp(0), xt) if prev_t >= 0 else torch.tensor(1.0)
        sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
        
        # prev_unkown part compute x_{t-1}
        noise = torch.randn_like(xt) 
        direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
        mask_0 = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
        prev_unknown_part = torch.sqrt(alphas_prev) * x0 + direction + mask_0 * sigmas * noise
        
        # known part compute x_{t-1} from x_0 of known part
        # prev_known_part = self.q_sample(original_image, prev_t, noise) 
        prev_known_part = torch.sqrt(alphas_prev) * original_image + torch.sqrt(1 - alphas_prev) * noise
        
        # combine known and unknown part
        pred = mask * prev_unknown_part + (1 - mask) * prev_known_part
        
        return pred, x0
    
    def set_timesteps(self, num_inference_steps, middle_time_steps, jump_length, jump_n_sample, add_small_steps=True):
        
        num_inference_steps = min(num_inference_steps, self.num_timesteps)
        self.num_inference_steps = num_inference_steps
        self.middle_time_steps = middle_time_steps
        timesteps = []
        
        jumps = {}
        for j in range(0, num_inference_steps - jump_length, jump_length):
            jumps[j] = jump_n_sample - 1

        t = num_inference_steps
        while t >= 1:
            t = t - 1
            timesteps.append(t)

            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(jump_length):
                    t = t + 1
                    timesteps.append(t)

        timesteps = np.array(timesteps) * (self.middle_time_steps // self.num_inference_steps)
        timesteps = timesteps[:-1] # 0 제외 
        
        if add_small_steps :
            while timesteps[-1] >= 2 : # 자잘한 값 추가 
                timesteps=np.append(timesteps,timesteps[-1] // 2) # 중간값 추가
        timesteps=np.append(timesteps, 0)
        
        if middle_time_steps == 1000:
            timesteps = np.insert(timesteps, 0, 999)
        else:
            timesteps = np.insert(timesteps, 0, middle_time_steps)
        self.timesteps = torch.from_numpy(timesteps).to(self.device)
    
    def apply_distance_transform(self,mask, max_distance=10):
        """
        5D 이진 마스크에 Distance Transform을 적용하여 부드럽게 확장된 마스크를 생성합니다.

        Args:
            mask (torch.Tensor): 0과 1로 이루어진 (B, C, F, H, W) 형태의 마스크 텐서.
            max_distance (float): 거리 최대값. 이 값을 넘어가는 거리들은 0으로 클리핑됩니다.

        Returns:
            torch.Tensor: Distance Transform이 적용된 PyTorch 텐서.
        """
        # 입력 검증: 5D 텐서인지 확인
        assert mask.dim() == 5, "Input mask must be a 5D tensor (B, C, F, H, W)"
        
        # 텐서를 넘파이 배열로 변환
        mask_np = mask.cpu().numpy().astype(np.uint8)

        # Distance Transform을 각 (H, W) 슬라이스에 대해 수행
        distance = np.zeros_like(mask_np, dtype=np.float32)
        for b in range(mask_np.shape[0]):  # 배치 차원
            for c in range(mask_np.shape[1]):  # 채널 차원
                for f in range(mask_np.shape[2]):  # 프레임 차원
                    distance[b, c, f] = distance_transform_edt(mask_np[b, c, f] == 0)

        # 거리 값을 max_distance를 기준으로 1과 0 사이의 범위로 클리핑
        soft_mask = np.clip(1 - distance / max_distance, 0, 1)

        # PyTorch 텐서로 변환하여 반환
        return torch.from_numpy(soft_mask).to(mask.device).to(torch.float32)

    ############################################################################################
    
    @torch.no_grad()
    def ddim_reverse_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
              _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            torch.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps), xt)
        
        # reverse sample
        mu = torch.sqrt(alphas_next) * x0 + torch.sqrt(1 - alphas_next) * eps
        return mu, x0
    
    @torch.no_grad()
    def ddim_reverse_sample_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20):
        # prepare input
        b = x0.size(0)
        xt = x0

        # reconstruction steps
        steps = torch.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        for step in steps:
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt
    


    
    
    
    
    
    @torch.no_grad()
    def plms_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
            - condition_fn: for classifier-based guidance (guided-diffusion).
            - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                      _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                     _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps
        
        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                 _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps
            
            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = torch.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
            xt_1 = torch.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0
        
        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    @torch.no_grad()
    def plms_sample_loop(self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None, plms_timesteps=20):
        # prepare input
        b = noise.size(0)
        xt = noise

        # diffusion process
        steps = (1 + torch.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps)).clamp(0, self.num_timesteps - 1).flip(0)
        eps_cache = []
        for step in steps:
            # PLMS sampling step
            t = torch.full((b, ), step, dtype=torch.long, device=xt.device)
            xt, _, eps = self.plms_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache)
            
            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def loss(self, x0, t, step, model, autoencoder, rank, model_kwargs={}, gs_data=None, noise=None, weight = None, use_div_loss= False):

        # noise = torch.randn_like(x0) if noise is None else noise # [80, 4, 8, 32, 32]
        noise = self.sample_loss(x0, noise)

        xt = self.q_sample(x0, t, noise=noise)

        # compute loss
        if self.loss_type in ['kl', 'rescaled_kl']:
            loss, _ = self.variational_lower_bound(x0, xt, t, model, model_kwargs)
            if self.loss_type == 'rescaled_kl':
                loss = loss * self.num_timesteps
        elif self.loss_type in ['mse', 'rescaled_mse', 'l1', 'rescaled_l1']: # self.loss_type: mse
            if model.module.use_lgm_refine:
                out = model(xt, self._scale_timesteps(t), x0=x0, 
                                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod, 
                                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                                sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                autoencoder=autoencoder,
                                **model_kwargs)
            else:
                out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']: # self.var_type: 'fixed_small'
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0

            if model.module.use_lgm_refine:
                loss = out['loss']
                # print("[Training PSNR]:", out['psnr'], "[Train Time]:", t)
            else:
                # MSE/L1 for x0/eps
                # target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
                target = {
                    'eps': noise, 
                    'x0': x0, 
                    'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0], 
                    'v':_i(self.sqrt_alphas_cumprod, t, xt) * noise - _i(self.sqrt_one_minus_alphas_cumprod, t, xt) * x0}[self.mean_type]
                loss = (out - target).pow(1 if self.loss_type.endswith('l1') else 2).abs().flatten(1).mean(dim=1)

            if weight is not None:
                loss = loss*weight   

            # div loss
            if use_div_loss and self.mean_type == 'eps' and x0.shape[2]>1:
                # derive  x0
                x0_ = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - \
                    _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out

                # # derive xt_1, set eta=0 as ddim
                # alphas_prev = _i(self.alphas_cumprod, (t - 1).clamp(0), xt)
                # direction = torch.sqrt(1 - alphas_prev) * out
                # xt_1 = torch.sqrt(alphas_prev) * x0_ + direction

                # ncfhw, std on f
                div_loss = 0.001/(x0_.std(dim=2).flatten(1).mean(dim=1)+1e-4)
                # print(div_loss,loss)
                loss = loss+div_loss

            # total loss
            loss = loss + loss_vlb
        elif self.loss_type in ['charbonnier']:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)

            # VLB for variation
            loss_vlb = 0.0
            if self.var_type in ['learned', 'learned_range']:
                out, var = out.chunk(2, dim=1)
                frozen = torch.cat([out.detach(), var], dim=1)  # learn var without affecting the prediction of mean
                loss_vlb, _ = self.variational_lower_bound(x0, xt, t, model=lambda *args, **kwargs: frozen)
                if self.loss_type.startswith('rescaled_'):
                    loss_vlb = loss_vlb * self.num_timesteps / 1000.0
            
            # MSE/L1 for x0/eps
            target = {'eps': noise, 'x0': x0, 'x_{t-1}': self.q_posterior_mean_variance(x0, xt, t)[0]}[self.mean_type]
            loss = torch.sqrt((out - target)**2 + self.epsilon)
            if weight is not None:
                loss = loss*weight
            loss = loss.flatten(1).mean(dim=1)
            
            # total loss
            loss = loss + loss_vlb
        # print(loss.shape)
        return loss

    def variational_lower_bound(self, x0, xt, t, model, model_kwargs={}, clamp=None, percentile=None):
        # compute groundtruth and predicted distributions
        mu1, _, log_var1 = self.q_posterior_mean_variance(x0, xt, t)
        mu2, _, log_var2, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile)

        # compute KL loss
        kl = kl_divergence(mu1, log_var1, mu2, log_var2)
        kl = kl.flatten(1).mean(dim=1) / math.log(2.0)
        
        # compute discretized NLL loss (for p(x0 | x1) only)
        nll = -discretized_gaussian_log_likelihood(x0, mean=mu2, log_scale=0.5 * log_var2)
        nll = nll.flatten(1).mean(dim=1) / math.log(2.0)

        # NLL for p(x0 | x1) and KL otherwise
        vlb = torch.where(t == 0, nll, kl)
        return vlb, x0
    
    @torch.no_grad()
    def variational_lower_bound_loop(self, x0, model, model_kwargs={}, clamp=None, percentile=None):
        r"""Compute the entire variational lower bound, measured in bits-per-dim.
        """
        # prepare input and output
        b = x0.size(0)
        metrics = {'vlb': [], 'mse': [], 'x0_mse': []}

        # loop
        for step in torch.arange(self.num_timesteps).flip(0):
            # compute VLB
            t = torch.full((b, ), step, dtype=torch.long, device=x0.device)
            # noise = torch.randn_like(x0)
            noise = self.sample_loss(x0)
            xt = self.q_sample(x0, t, noise)
            vlb, pred_x0 = self.variational_lower_bound(x0, xt, t, model, model_kwargs, clamp, percentile)

            # predict eps from x0
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / \
                  _i(self.sqrt_recipm1_alphas_cumprod, t, xt)

            # collect metrics
            metrics['vlb'].append(vlb)
            metrics['x0_mse'].append((pred_x0 - x0).square().flatten(1).mean(dim=1))
            metrics['mse'].append((eps - noise).square().flatten(1).mean(dim=1))
        metrics = {k: torch.stack(v, dim=1) for k, v in metrics.items()}

        # compute the prior KL term for VLB, measured in bits-per-dim
        mu, _, log_var = self.q_mean_variance(x0, t)
        kl_prior = kl_divergence(mu, log_var, torch.zeros_like(mu), torch.zeros_like(log_var))
        kl_prior = kl_prior.flatten(1).mean(dim=1) / math.log(2.0)

        # update metrics
        metrics['prior_bits_per_dim'] = kl_prior
        metrics['total_bits_per_dim'] = metrics['vlb'].sum(dim=1) + kl_prior
        return metrics

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
        #return t.float()
