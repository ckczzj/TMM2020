import torch
import torch.nn as nn
import torch.nn.functional as F
from .bi_lstm import BiLSTM


class Coordinate(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm=BiLSTM(input_size,hidden_size)
        self.start_linear1=nn.Linear(2*hidden_size,hidden_size,bias=True)
        self.start_linear2=nn.Linear(hidden_size,1,bias=False)
        self.end_linear1=nn.Linear(2*hidden_size,hidden_size,bias=True)
        self.end_linear2=nn.Linear(hidden_size,1,bias=False)

        self.perceptron=nn.Linear(2*hidden_size,1,bias=True)

    # fusion: batch_size * video_len * input_size
    # video_mask: batch_size * video_len
    def forward(self, fusion, video_mask):
        video_len=video_mask.sum(dim=1)

        # batch_size * video_len * (2 * hidden_size)
        context=self.lstm(fusion,video_len)

        # batch_size * video_len * hidden_size
        start_tmp=torch.tanh(self.start_linear1(context))
        start_tmp=start_tmp*video_mask.unsqueeze(2)
        # batch_size * video_len
        start=self.start_linear2(start_tmp).squeeze(-1)
        start=start.masked_fill(video_mask == 0, -1e30)
        # batch_size * video_len
        start_score=F.softmax(start,dim=1)
        start_score=start_score*video_mask

        # batch_size * video_len * hidden_size
        end_tmp=torch.tanh(self.end_linear1(context))
        end_tmp=end_tmp*video_mask.unsqueeze(2)
        # batch_size * video_len
        end=self.end_linear2(end_tmp).squeeze(-1)
        end=end.masked_fill(video_mask == 0, -1e30)
        # batch_size * video_len
        end_score=F.softmax(end,dim=1)
        end_score=end_score*video_mask

        # batch_size * video_len
        p=torch.sigmoid(self.perceptron(context).squeeze(-1))

        # batch_size * video_len
        p=p*video_mask

        return p,start_score,end_score,torch.argmax(start_score,dim=1),torch.argmax(end_score,dim=1)
