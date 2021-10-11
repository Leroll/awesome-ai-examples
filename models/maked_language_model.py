from collections import OrderedDict
from torch import nn


class MaskedLanguageModel(nn.Module):
    """mlm, pet_mlm 均可
    """
    def __init__(self,
                 name: str,
                 device: str,
                 pretrain_model
                 ):
        """
        args：
            name： 模型名字
            device： 'cpu' or 'cuda:0'
            pretrain_model: pretrain_model, 抱抱脸格式
        """
        super().__init__()
        self.name = name
        self.device = device
        self.pretrain_model = pretrain_model
        self.final_part = self.__init_layers()

        self.to(self.device)  # 需要放在最后，等所有weight都初始化后再更改device

    def __init_layers(self):
        hidden_size = self.pretrain_model.config.hidden_size
        embedding_size = self.pretrain_model.embeddings.word_embeddings.num_embeddings
        final_part = nn.Sequential(OrderedDict([
            ('final_Linear',
             nn.Linear(hidden_size, hidden_size, bias=True)),
            ('final_layernorm',
             nn.LayerNorm(hidden_size, eps=1e-12)),
            ('final_embedding',
             nn.Linear(hidden_size, embedding_size, bias=False))
        ]))
        return final_part

    def forward(self, x):
        """x为pair对tokenize 好之后的token_id
        """
        x = self.pretrain_model(**x)[0]  # hidden
        y_pre = self.final_part(x)

        return y_pre











