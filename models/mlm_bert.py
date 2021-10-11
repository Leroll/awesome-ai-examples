import torch
from torch import nn

from pretrain_train_base_model import PretrainBasedModels


class MlmBert(PretrainBasedModels):
    """用于 mlm 训练
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        hidden_size = self.pretrain_model.config.hidden_size
        embedding_size = self.pretrain_model.embeddings.word_embeddings.num_embeddings
        self.final_part = nn.Sequential(OrderedDict([
            ('final_Linear',
             nn.Linear(hidden_size, hidden_size, bias=True)),
            ('final_layernorm',
             nn.LayerNorm(hidden_size, eps=1e-12)),
            ('final_embedding',
             nn.Linear(hidden_size, embedding_size, bias=False))
        ]))

        # 因为emb的w本身就是v*d的, 这里不需要转置
        embedding_p = [p for p in self.pretrain_model.embeddings.word_embeddings.parameters()][0]
        self.final_part.final_embedding.weight.data = embedding_p

    def forward(self, x):
        token = self.get_token_from_single(x, is_split_into_words=False)
        x = self.pretrain_model(**token)[0]
        y_pre = self.final_part(x)

        return y_pre

    def get_y_mask(self, y):
        """
        like attention_mask, get y_mask from y_token
        """
        y_mask = []
        for i in y:
            temp_mask = []
            for j in i:
                m = 0 if j in [0, 101, 102] else 1
                temp_mask.append(m)
            y_mask.append(temp_mask)

        y_mask = torch.tensor(y_mask).to(self.device)
        return y_mask