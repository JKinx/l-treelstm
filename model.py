import torch
from torch import nn
from tree_lstm import BTreeLSTMParser, BTreeLSTMComposer

class Model(nn.Module):
    def __init__(self, vocab_size, idim, hdim, p_tdim, c_tdim, odim, gumbel_temperature):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, idim)
        self.parser = BTreeLSTMParser(idim, hdim, p_tdim, gumbel_temperature)
        self.tree_embeddings = nn.Embedding(vocab_size, idim)
        self.composer = BTreeLSTMComposer(idim, hdim, c_tdim)
        self.linear = nn.Linear(hdim, odim)
        
        self.running_reward_var = 1.0
        self.norm_alpha = 0.9
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.word_embeddings.weight, 0.0, 0.01)
        nn.init.normal_(self.tree_embeddings.weight, 0.0, 0.01)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, val=0)
        self.parser.reset_parameters()
        self.composer.reset_parameters()
        
    def single_pass(self, x, mask, labels):
        tree_compositions, log_probs, entropy, norm_entropy = self.parser(self.word_embeddings(x), mask)
        out = self.composer(self.tree_embeddings(x), mask, tree_compositions)
        logits = self.linear(out)
        rewards = self.criterion(logits, labels)
        return tree_compositions, log_probs, entropy, norm_entropy, logits, rewards
    
    def get_baseline(self, x, mask, labels):
        with torch.no_grad():
            self.eval()
            rewards_c = self.single_pass(x, mask, labels)[-1]
            self.train()
            return rewards_c
    
    def normalize_rewards(self, rewards):
        with torch.no_grad():
            self.running_reward_var = self.norm_alpha * self.running_reward_var + \
                                        (1 - self.norm_alpha) * rewards.var()
            return rewards / self.running_reward_var.sqrt().clamp(min=1.0)
         
    def forward(self, x, mask, labels):
        tree_compositions, log_probs, entropy, norm_entropy, logits, rewards =  self.single_pass(x, mask, labels)
        loss = rewards.mean()
        if self.training:
            baseline = self.get_baseline(x, mask, labels)
            rewards = self.normalize_rewards(rewards - baseline)
        predictions = logits.argmax(dim=-1)
        return predictions, tree_compositions, loss, rewards.detach(), log_probs, entropy, norm_entropy
    
    def evaluate(self, x, mask, eval_tree_compositions):
        _, log_probs, _, norm_entropy = self.parser(self.word_embeddings(x), mask, eval_tree_compositions)
        return log_probs, norm_entropy