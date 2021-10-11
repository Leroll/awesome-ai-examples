import torch
from torch import nn

from pretrain_train_base_model import PretrainBasedModels

class SentenceBert(PretrainBasedModels):
    """
    args:
        pretrain_model: 需要是bert
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_size = self.pretrain_model.config.hidden_size
        self.linear = nn.Linear(hidden_size * 3, 2)

    def forward(self, q1, q2):
        q = []
        for temp_q in [q1, q2]:
            temp_q = self.get_token_from_single(temp_q, is_split_into_words=False)
            temp_q = self._through_bert_then_mean(temp_q, mean_mode='attention_mask')
            q.append(temp_q)

        diff = torch.abs(q[0] - q[1])
        h = torch.cat((q[0], q[1], diff), dim=-1)
        h = self.linear(h)
        return h
