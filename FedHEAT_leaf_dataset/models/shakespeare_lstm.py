import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY


@ENCODER_REGISTRY.register()
class Shakespeare_LSTM(nn.Module):
    def __init__(
        self,
        args,
        num_classes,
        vocab_size=None,
        embed_dim=8,
        hidden_dim=256,
        num_layers=2,
        l2_norm=False,
        **kwargs,
    ):
        super(Shakespeare_LSTM, self).__init__()

        self.vocab_size = num_classes if vocab_size is None else vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=None,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.hidden_to_embed = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

        self.num_layers = num_layers
        self.l2_norm = l2_norm

    def forward(self, x, mlb_level=None):
        """
        x: LongTensor of shape [B, T]
           each entry is a character index
        """

        emb = self.embedding(x)          # [B, T, embed_dim]
        layer0 = emb

        out, (h_n, c_n) = self.lstm(emb) # out: [B, T, hidden_dim]
        layer1 = out

        feature = out[:, -1, :]          # [B, hidden_dim]
        if self.l2_norm:
            feature = F.normalize(feature, p=2, dim=1)

        projected = self.hidden_to_embed(feature)                    # [B, embed_dim]
        logit = projected @ self.embedding.weight.transpose(0, 1)    # [B, vocab_size]
        logit = logit + self.output_bias

        return {
            "layer0": layer0,
            "layer1": layer1,
            "feature": feature,
            "logit": logit,
        }