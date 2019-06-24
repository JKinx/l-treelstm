import torch
from torch import nn
import torch.distributions as ds
from torch.distributions import utils as distr_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions.categorical import Categorical as TorchCategorical
from functools import partial

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

def gumbel_softmax(logits, mask=None):
    with torch.no_grad():
        epsilon = 1e-20

        # get gumbel noise
        unif = ds.Uniform(0,1).sample(logits.size())
        gumbel_noise = -(-(unif + epsilon).log() + epsilon).log().to(logits)

        # get samples 
        new_logits = logits + gumbel_noise
        if mask is None:
            y = new_logits.softmax(dim=-1)
        else:
            y = masked_softmax(new_logits, mask)
        
        # hard samples
        y = torch.zeros_like(y).scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0).to(logits)
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

def get_lr_scheduler(optimizer, mode='max', factor=0.5, patience=10, threshold=1e-4, threshold_mode='rel'):
    def reduce_lr(self, epoch):
        ReduceLROnPlateau._reduce_lr(self, epoch)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode, factor, patience, False, threshold, threshold_mode)
    lr_scheduler._reduce_lr = partial(reduce_lr, lr_scheduler)
    return lr_scheduler

class EarlyStopping:
    def __init__(self, mode='max', patience=20, threshold=1e-4, threshold_mode='rel'):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.num_bad_epochs = 0
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.last_epoch = -1
        self.is_converged = False
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self.best = self.mode_worse

    def is_improved(self):
        return self.num_bad_epochs == 0

    def step(self, metrics):
        if self.is_converged:
            raise ValueError
        current = metrics
        self.last_epoch += 1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.is_converged = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon
        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold
        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = (-float('inf'))

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)
        
class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
class Categorical:
    def __init__(self, scores, mask=None):
        self.mask = mask
        if mask is None:
            self.cat_distr = TorchCategorical(F.softmax(scores, dim=-1))
            self.n = scores.shape[0]
            self.log_n = math.log(self.n)
        else:
            self.n = self.mask.sum(dim=-1)
            self.log_n = (self.n + 1e-17).log()
            self.cat_distr = TorchCategorical(Categorical.masked_softmax(scores, self.mask))

    @lazy_property
    def probs(self):
        return self.cat_distr.probs

    @lazy_property
    def logits(self):
        return self.cat_distr.logits

    @lazy_property
    def entropy(self):
        if self.mask is None:
            return self.cat_distr.entropy() * (self.n != 1)
        else:
            entropy = - torch.sum(self.cat_distr.logits * self.cat_distr.probs * self.mask, dim=-1)
            does_not_have_one_category = (self.n != 1.0).to(dtype=torch.float32)
            # to make sure that the entropy is precisely zero when there is only one category
            return entropy * does_not_have_one_category

    @lazy_property
    def normalized_entropy(self):
        return self.entropy / (self.log_n + 1e-17)

    def rsample(self, temperature=None, gumbel_noise=None):
        with torch.no_grad():
            uniforms = torch.empty_like(self.probs).uniform_()
            uniforms = distr_utils.clamp_probs(uniforms)
            gumbel_noise = -(-uniforms.log()).log()
            # TODO(serhii): This is used for debugging (to get the same samples) and is not differentiable.
            # gumbel_noise = None
            # _sample = self.cat_distr.sample()
            # sample = torch.zeros_like(self.probs)
            # sample.scatter_(-1, _sample[:, None], 1.0)
            # return sample, gumbel_noise

        with torch.no_grad():
            scores = (self.logits + gumbel_noise)
            scores = Categorical.masked_softmax(scores, self.mask)
            sample = torch.zeros_like(scores)
            sample.scatter_(-1, scores.argmax(dim=-1, keepdim=True), 1.0)
            return sample

    def log_prob(self, value):
        if value.dtype == torch.long:
            if self.mask is None:
                return self.cat_distr.log_prob(value)
            else:
                return self.cat_distr.log_prob(value) * (self.n != 0.).to(dtype=torch.float32)
        else:
            max_values, mv_idxs = value.max(dim=-1)
            relaxed = (max_values - torch.ones_like(max_values)).sum().item() != 0.0
            if relaxed:
                raise ValueError("The log_prob can't be calculated for the relaxed sample!")
            return self.cat_distr.log_prob(mv_idxs) * (self.n != 0.).to(dtype=torch.float32)

    @staticmethod
    def masked_softmax(logits, mask):
        """
        This method will return valid probability distribution for the particular instance if its corresponding row
        in the `mask` matrix is not a zero vector. Otherwise, a uniform distribution will be returned.
        This is just a technical workaround that allows `Categorical` class usage.
        If probs doesn't sum to one there will be an exception during sampling.
        """
        probs = F.softmax(logits, dim=-1) * mask
        probs = probs + (mask.sum(dim=-1, keepdim=True) == 0.).to(dtype=torch.float32)
        Z = probs.sum(dim=-1, keepdim=True)
        return probs / Z