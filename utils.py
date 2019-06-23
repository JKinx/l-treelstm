import torch
from torch import nn
import torch.distributions as ds

def reset_lstm(lstm):
    for parameter in lstm.named_parameters():
        name = parameter[0]
        if "bias" in name:
            nn.init.constant_(parameter[1], val=5)
        elif "ih" in name:
            nn.init.xavier_uniform_(parameter[1])
        elif "hh" in name:
            nn.init.orthogonal_(parameter[1])
        else:
            raise ValueError("Problem")

def masked_softmax(logits, mask):
    probs = torch.softmax(logits, dim=-1) * mask
    probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)
    Z = probs.sum(dim=-1, keepdim=True)
    return probs / Z

def gumbel_softmax(logits, temperature, mask=None):
    epsilon = 1e-20
    
    # get gumbel noise
    unif = ds.Uniform(0,1).sample(logits.size())
    gumbel_noise = -(-(unif + epsilon).log() + epsilon).log()
    
    # get samples 
    new_logits = (logits + gumbel_noise) / temperature
    if mask is None:
        y = new_logits.softmax(dim=-1)
    else:
        y = masked_softmax(new_logits, mask)
        
    # hard samples
    y_st = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
    # sample with gradients
    y = (y_st - y).detach() + y
    return y

def cat_entropy(logits, mask):
    probs = masked_softmax(logits, mask) + 1e-17
    entropy = -(probs.log() * probs * mask).sum(-1) * (mask.sum(-1) != 1.).float()
    return entropy

def cat_norm_entropy(logits, mask):
    log_n = (mask.sum(-1) + 1e-17).log()
    entropy = cat_entropy(logits, mask)
    return entropy / (log_n + 1e-17)

def cat_logprob(logits, mask, values):
    # values is one-hot encoded
    lprobs = masked_softmax(logits, mask).log()
    log_prob = torch.gather(lprobs, -1, values.argmax(-1, keepdim=True)).squeeze()
    return log_prob * (mask.sum(-1) != 0.).float()

def get_seqmask(seqlens):
    # get sequence mask from seqlens
    # output shape = batch X maxlen
    maxlen = seqlens.max()
    batch = seqlens.shape[0]
    arange = torch.arange(maxlen).unsqueeze(0).expand(batch, -1).long().to(seqlens)
    return (arange < seqlens.unsqueeze(-1)).float()