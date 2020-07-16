import torch
import torch.nn as nn


class TanhAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w1 = nn.Linear(d, d, bias=True)
        self.w2 = nn.Linear(d, d, bias=False)
        self.wt = nn.Linear(d, 1, bias=False)

    # video_feature: batch_size * video_len * d
    # video_mask: batch_size * video_len
    # sentence_feature: batch_size * sentence_len * d
    # sentence_mask: batch_size * sentence_len
    def forward(self, video_feature, video_mask, sentence_feature, sentence_mask):
        # batch_size * video_len * d
        tmp1 = self.w1(video_feature)
        # batch_size * sentence_len * d
        tmp2 = self.w2(sentence_feature)

        # remove
        tmp1=tmp1*video_mask.unsqueeze(2)
        tmp2=tmp2*sentence_mask.unsqueeze(2)
        # remove

        # batch_size * video_len * sentence_len * d
        tmp = tmp1.unsqueeze(2) + tmp2.unsqueeze(1)

        # remove
        mask=(video_mask.unsqueeze(2)*sentence_mask.unsqueeze(1)).unsqueeze(-1)
        tmp=tmp*mask
        # remove

        # batch_size * video_len * sentence_len
        M=self.wt(torch.tanh(tmp)).squeeze(-1)

        M=M*(video_mask.unsqueeze(2)*sentence_mask.unsqueeze(1))

        return M

        # print(self.wt(tmp[0][110][4]))
        # print(self.wt((self.w1(video[0][110])+self.w2(sentence[0][4]))))

        # if memory_mask is not None:
        #     memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
        #     S = S.masked_fill(memory_mask == 0, -1e30)
        #     # for forward, backward, S: [nb, len, len]
        #     if self.direction == 'forward':
        #         length = S.size(1)
        #         forward_mask = torch.ones(length, length)
        #         for i in range(1, length):
        #             forward_mask[i, 0:i] = 0
        #         S = S.masked_fill(forward_mask.cuda().unsqueeze(0) == 0, -1e30)
        #     elif self.direction == 'backward':
        #         length = S.size(1)
        #         backward_mask = torch.zeros(length, length)
        #         for i in range(0, length):
        #             backward_mask[i, 0:i + 1] = 1
        #         S = S.masked_fill(backward_mask.cuda().unsqueeze(0) == 0, -1e30)
        # S = self.dropout(F.softmax(S, -1))
        # return torch.matmul(S, memory)  # [nb, len1, d]
