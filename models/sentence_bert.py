import torch
from torch import nn

class SentenceBert(nn.Module):
    """
    args:
        pretrain_model: 需要是bert
    """

    def __init__(self,
                 name: str,
                 device: str,
                 pretrain_model):

        super().__init__()
        self.name = name
        self.device = device
        self.pretrain_model = pretrain_model

        self.final_layer = self.__init_final_layer()

    def __init_final_layer(self):
        hidden_size = self.pretrain_model.config.hidden_size
        final_layer = nn.Linear(hidden_size * 3, 2)
        return final_layer

    def forward(self, inputs):
        """
        args:
            inputs: tokenizer之后的字典

        ## TODO 原始的输入里面还家了q1， q2，这个部分需要放到transdata里面去
        ## test
        """

        q = []
        for temp_q in [q1, q2]:
            temp_q = self.get_token_from_single(temp_q, is_split_into_words=False)
            temp_q = self._through_bert_then_mean(temp_q, mean_mode='attention_mask')
            q.append(temp_q)

        diff = torch.abs(q[0] - q[1])
        h = torch.cat((q[0], q[1], diff), dim=-1)
        h = self.linear(h)
        return h
