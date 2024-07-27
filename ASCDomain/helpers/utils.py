import numpy as np
import torch
from torch.distributions.beta import Beta

def NAME_TO_WIDTH(name):
    mn_map = {
        'mn01': 0.1,
        'mn02': 0.2,
        'mn04': 0.4,
        'mn05': 0.5,
        'mn06': 0.6,
        'mn08': 0.8,
        'mn10': 1.0,
        'mn12': 1.2,
        'mn14': 1.4,
        'mn16': 1.6,
        'mn20': 2.0,
        'mn30': 3.0,
        'mn40': 4.0,
    }

    dymn_map = {
        'dymn04': 0.4,
        'dymn10': 1.0,
        'dymn20': 2.0
    }

    try:
        if name.startswith('dymn'):
            w = dymn_map[name[:6]]
        else:
            w = mn_map[name[:4]]
    except:
        w = 1.0

    return w

def mixstyle(x, p=0.4, alpha=0.3, eps=1e-6):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)
    # original: batch x channel x frequency x time -> mean(frequency, time)
    # freq_style: batch x channel x frequency x time -> mean(channel, time)
    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True) 
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input

    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling

    lmda = Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)  # sample instance-wise convex weights
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed frequency statistics
    return x