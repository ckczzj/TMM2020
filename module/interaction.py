import torch
import torch.nn as nn
import torch.nn.functional as F
from .tanh_attention import TanhAttention


class Interaction(nn.Module):
    # d==2*hidden_size==512
    def __init__(self,d,dropout=0.0):
        super().__init__()
        self.d=d
        self.dropout = nn.Dropout(dropout)
        self.self_attention=TanhAttention(d=d)
        # self.interaction=nn.Linear(2*d,2*d,bias=True)
        self.gate=nn.Linear(2*d,1,bias=True)
    # video_feature: batch_size * video_len * d
    # video_mask: batch_size * video_len
    # sentence_feature: batch_size * sentence_len * d
    # sentence_mask: batch_size * sentence_len
    # M: batch_size * video_len * sentence_len
    def forward(self, video_feature, video_mask, sentence_feature, sentence_mask):
        # batch_size * video_len * sentence_len
        M=self.self_attention(video_feature,video_mask,sentence_feature,sentence_mask)

        # batch_size * video_len * sentence_len
        row = M.masked_fill(sentence_mask.unsqueeze(1) == 0, -1e30)
        col = M.masked_fill(video_mask.unsqueeze(2) == 0, -1e30)

        # batch_size * video_len * sentence_len
        M_row=self.dropout(F.softmax(row,dim=2))
        M_col=self.dropout(F.softmax(col,dim=1))

        # remove
        mask=video_mask.unsqueeze(2)*sentence_mask.unsqueeze(1)
        M_row=M_row*mask
        M_col=M_col*mask
        # remove

        # batch_size * video_len * video_len
        D=torch.matmul(M_row,M_col.transpose(1,2))
        # batch_size * sentence_len * sentence_len
        # L=torch.matmul(M_col.transpose(1,2),M_row)

        # remove
        D_mask=video_mask.unsqueeze(2)*video_mask.unsqueeze(1)
        D=D*D_mask
        # L_mask=sentence_mask.unsqueeze(2)*sentence_mask.unsqueeze(1)
        # L=L*L_mask
        # remove

        # print(D.shape)
        # print(L,L.shape)
        # batch_size * video_len * d
        text_guided_self_attention=torch.matmul(D,video_feature)
        # batch_size * sentence_len * d
        # video_guided_self_attention=torch.matmul(L,sentence_feature)

        # remove
        text_guided_self_attention=text_guided_self_attention*video_mask.unsqueeze(2)
        # video_guided_self_attention=video_guided_self_attention*sentence_mask.unsqueeze(2)
        # remove

        # batch_size * video_len * (2 * d)
        video_interaction=torch.cat((video_feature,text_guided_self_attention),dim=2)
        # batch_size * sentence_len * (2 * d)
        # sentence_interaction=torch.cat((sentence_feature,video_guided_self_attention),dim=2)

        # batch_size * video_len * (2 * d)
        video_interaction_gate=torch.sigmoid(self.gate(video_interaction))

        # batch_size * video_len * (2 * d)
        video_interaction=video_interaction*video_interaction_gate

        # remove
        video_interaction=video_interaction*video_mask.unsqueeze(2)
        # sentence_interaction=sentence_interaction*sentence_mask.unsqueeze(2)
        # remove

        # batch_size
        video_len=video_mask.sum(dim=1)
        # batch_size * sentence_len
        alignment=torch.sum(M,dim=1)/video_len.unsqueeze(-1)
        alignment=alignment.masked_fill(sentence_mask==0,-1e30)
        attention_score=F.softmax(alignment,dim=1)
        attention_score=attention_score*sentence_mask


        # batch_size * sentence_len * d
        sentence_feature=sentence_feature*attention_score.unsqueeze(-1)
        # batch_size * d
        sentence_feature=sentence_feature.sum(dim=1)

        b,vl,_=video_interaction.shape
        # batch_size * video_len * d
        sentence_feature=sentence_feature.unsqueeze(1).expand(b,vl,self.d)

        # batch_size * video_len * (3 * d)
        fusion=torch.cat((video_interaction,sentence_feature),dim=2)
        return fusion
