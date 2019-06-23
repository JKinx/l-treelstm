import torch
from torch import nn
import torch.distributions as ds
from utils import *

class BTreeLSTMCell(nn.Module):
    def __init__(self, hdim, dropout_prob=None):
        super().__init__()
        self.hdim = hdim
        self.linear = nn.Linear(in_features = 2*self.hdim, out_features = 5*self.hdim)
        if dropout_prob is None:
            self.dropout = lambda x : x
        else: 
            self.dropout = nn.Dropout(dropout_prob)  
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val = 0)
        nn.init.constant_(self.linear.bias[self.hdim:3*self.hdim], val = 1)
        
    def forward(self, hl, cl, hr, cr):
        # h[], c[] : Shape = batch X seqlen X hdim
        h = torch.cat([hl, hr], dim=-1)
        i, fl, fr, o, g = self.linear(h).chunk(chunks = 5, dim = -1)
        cp = self.dropout(g.tanh_()) * i.sigmoid_() + cl * fl.sigmoid_() + cr * fr.sigmoid_()
        hp = o.sigmoid() * cp.tanh()
        return hp, cp
    
class BTreeLSTMBase(nn.Module):
    def __init__(self, idim, hdim, tdim, dropout_prob=None):
        super().__init__()
        self.leaftransformer_lstm = nn.LSTM(idim, tdim)
        self.leaftransformer_linear = nn.Linear(tdim, 2*hdim)
        
        self.treelstm_cell = BTreeLSTMCell(hdim, dropout_prob)
        
        BTreeLSTMBase.reset_parameters(self)
    
    def reset_parameters(self):
        nn.init.orthogonal_(self.leaftransformer_linear.weight)
        nn.init.constant_(self.leaftransformer_linear.bias, val=0)
        self.treelstm_cell.reset_parameters()
        self.leaftransformer_lstm.reset_parameters()
    
    def transform_leafs(self, x):
        # x : Shape = batch X seqlen X idim
        x = self.leaftransformer_lstm(x)[0]
        # Shape = batch X seqlen X 2*hdim
        x = self.leaftransformer_linear(x).tanh()
        # Shape = (batch X seqlen X hdim, batch X seqlen X hdim)
        return x.chunk(chunks=2, dim=-1)
    
    def compose(self, composition, hl, cl, hr, cr, hp, cp, mask):
        # composition : Shape = batch X seqlen
        # hl, hr, hp, cl, cr, cp : Shape = batch X seqlen X hdim
        # mask : Shape = batch X seqlen
        # mask is for padding
        cumsum = torch.cumsum(composition, dim=-1)
        
        # Shape = batch X maxlen X 1
        # for broadcasting
        ml = (1 - cumsum).unsqueeze(-1)
        mr = (cumsum - 1).unsqueeze(-1)
        mask = mask.unsqueeze(-1)
        composition = composition.unsqueeze(-1)
        
        # next layer
        hp = mask * (ml * hl + mr * hr + composition * hp) + (1 - mask) * hl
        cp = mask * (ml * cl + mr * cr + composition * cp) + (1 - mask) * cl
        return hp, cp
    
    def forward(self, *inputs):
        raise NotImplementedError
        
class BTreeLSTMParser(BTreeLSTMBase):
    def __init__(self, idim, hdim, tdim, gumbel_temperature, dropout_prob=None):
        super().__init__(idim, hdim, tdim, dropout_prob)
        self.q = nn.Parameter(torch.FloatTensor(hdim))
        # temperature for gumbel softmax
        self.gumbel_temperature = gumbel_temperature
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.q, mean=0, std=0.01)
    
    def sample_composition(self, query_weights, mask):
        if self.training:
            # sample from gumbel_softmax if training
            composition = gumbel_softmax(query_weights, self.gumbel_temperature, mask)
        else:
            # greedy if not
            logits = masked_softmax(query_weights, mask)
            composition = torch.zeros_like(logits).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return composition
    
    def step(self, h, c, mask, eval_composition):
        # get left and right sides
        hl, hr = h[:,:-1], h[:,1:]
        cl, cr = c[:,:-1], c[:,1:]
        # composed states
        hp, cp = self.treelstm_cell(hl, cl, hr, cr)
        
        # get composition query weights
        query_weights = torch.matmul(hp, self.q)
        if eval_composition is None:
            # sample is not given
            composition = self.sample_composition(query_weights, mask)
        else:
            # use provided mergers if available
            composition = eval_composition
            
        # perform composition
        hp, cp = self.compose(composition, hl, cl, hr, cr, hp, cp, mask)
        return hp, cp, composition, query_weights
        
    def forward(self, x, mask, eval_tree_compositions=None):
        # transform the leafs
        h, c = self.transform_leafs(x)
        
        # values to record
        entropy = []
        norm_entropy = []
        log_probs = []
        tree_compositions = []
        hs = [h]
        cs = [c]
        for i in range(x.shape[1]-1):
            # get the relevant mask (1 less than the pervious one)
            rel_mask = mask[:, i+1:]
            # perfrom a step (move up a layer)
            eval_composition = None if eval_tree_compositions is None else eval_tree_compositions[i]
            h, c, composition, query_weights = self.step(h, c, rel_mask, eval_composition)
            tree_compositions.append(composition)
            entropy.append(cat_entropy(query_weights, rel_mask))
            norm_entropy.append(cat_norm_entropy(query_weights, rel_mask))
            log_probs.append(cat_logprob(query_weights, rel_mask, composition))
            hs.append(h)
            cs.append(c)
            
        entropy = sum(entropy)
        norm_entropy = sum(norm_entropy) / (mask[:, 2:].sum(-1) + 1e-17)
        log_probs = sum(log_probs)
        
        return tree_compositions, log_probs, entropy, norm_entropy
    
class BTreeLSTMComposer(BTreeLSTMBase):
    def __init__(self, idim, hdim, tdim, dropout_prob=None):
        super().__init__(idim, hdim, tdim, dropout_prob)
    
    def forward(self, x, mask, tree_compositions):        
        # transform the leafs
        h, c = self.transform_leafs(x)
        
        # perform merges
        for i in range(x.shape[1]-1):
            hl, hr = h[:,:-1], h[:,1:]
            cl, cr = c[:,:-1], c[:,1:]
            hp, cp = self.treelstm_cell(hl, cl, hr, cr)
            h, c = self.compose(tree_compositions[i], hl, cl, hr, cr, hp, cp, mask[:, i+1:])
        # return root
        return h.squeeze(1)
