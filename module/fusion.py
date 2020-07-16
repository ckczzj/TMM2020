import torch
import torch.nn as nn
import torch.nn.functional as F
from .tanh_attention import TanhAttention


class MultiModalFusion(nn.Module):
    # d==4*hidden_dim
    def __init__(self,d,output_dim,dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention=TanhAttention(d=d)
        self.linear=nn.Linear(4*d, output_dim, bias=True)

    # video_feature: batch_size * video_len * d
    # video_mask: batch_size * video_len
    # sentence_feature: batch_size * sentence_len * d
    # sentence_mask: batch_size * sentence_len
    def forward(self,video_feature, video_mask, sentence_feature, sentence_mask):

        # batch_size * video_len * sentence_len
        S=self.attention(video_feature,video_mask,sentence_feature,sentence_mask)

        # batch_size * video_len * sentence_len
        row = S.masked_fill(sentence_mask.unsqueeze(1) == 0, -1e30)

        # batch_size * video_len * sentence_len
        S_row=self.dropout(F.softmax(row,dim=2))

        # remove
        mask=video_mask.unsqueeze(2)*sentence_mask.unsqueeze(1)
        S_row=S_row*mask
        # remove

        # batch_size * video_len * d
        texual_representation=torch.matmul(S_row,sentence_feature)

        #remove
        texual_representation=texual_representation*video_mask.unsqueeze(-1)
        #remove

        concat=torch.cat((video_feature,texual_representation,video_feature*texual_representation,video_feature-texual_representation),dim=2)
        concat=concat*video_mask.unsqueeze(2)

        # batch_size * video_len * output_dim
        fusion=torch.tanh(self.linear(concat))

        # remove
        fusion=fusion*video_mask.unsqueeze(-1)
        # remove

        return fusion



