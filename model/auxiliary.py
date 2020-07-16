import torch
import torch.nn as nn
import torch.nn.functional as F

class Auxiliary(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        # self.video_norm=nn.BatchNorm1d(num_features=2*self.config["hidden_dim)
        # self.sentence_norm=nn.BatchNorm1d(num_features=2*self.config["hidden_dim)
        self.self_attention_alignment=nn.Linear(2*self.config["hidden_dim"],1,bias=False)
        self.fusion=nn.Linear(4*2*self.config["hidden_dim"], 2*2*self.config["hidden_dim"], bias=True)
        self.linear1=nn.Linear(2*2*self.config["hidden_dim"],2*self.config["hidden_dim"],bias=True)
        self.linear2=nn.Linear(2*self.config["hidden_dim"],1,bias=True)

    # video_feature: batch_size * video_len * (2 * hidden_dim)
    # video_mask: batch_size * video_len
    # sentence_feature: batch_size * sentence_len * (2 * hidden_dim)
    # sentence_mask: batch_size * sentence_len
    # p: batch_size * video_len 已经乘过video_mask
    # gt: batch_size * video_len
    def forward(self,video_feature,video_mask,sentence_feature,sentence_mask,p,gt):

        # video_feature=self.video_norm(video_feature.transpose(1,2)).transpose(1,2)
        #
        # sentence_feature=self.sentence_norm(sentence_feature.transpose(1,2)).transpose(1,2)

        # batch_size * sentence_len
        attention_score=self.self_attention_alignment(sentence_feature).squeeze(-1)
        attention_score=attention_score.masked_fill(sentence_mask == 0, -1e30)
        attention_score=F.softmax(attention_score,dim=1)
        attention_score=attention_score*sentence_mask

        # batch_size * sentence_len * (2 * hidden_dim)
        global_sentence_feature=attention_score.unsqueeze(-1)*sentence_feature

        # batch_size * (2 * hidden_dim)
        global_sentence_feature=global_sentence_feature.sum(dim=1)


        # batch_size * (2 * hidden_dim)
        f_global_video_feature=(video_feature*p.unsqueeze(-1)).sum(dim=1)
        gt_global_video_feature=(video_feature*gt.unsqueeze(-1)).sum(dim=1)

        # batch_size * (2 * hidden_dim)
        f_global_video_feature=f_global_video_feature/(p.sum(dim=1).unsqueeze(-1))
        gt_global_video_feature=gt_global_video_feature/(gt.sum(dim=1).unsqueeze(-1))


        # batch_size * (2 * hidden_dim)
        f_fusion=torch.tanh(self.fusion(torch.cat((global_sentence_feature,f_global_video_feature,global_sentence_feature*f_global_video_feature,global_sentence_feature-f_global_video_feature),dim=1)))
        gt_fusion=torch.tanh(self.fusion(torch.cat((global_sentence_feature,gt_global_video_feature,global_sentence_feature*gt_global_video_feature,global_sentence_feature-gt_global_video_feature),dim=1)))
        # f_fusion=torch.tanh(self.fusion(torch.cat((global_sentence_feature,f_global_video_feature),dim=1)))
        # gt_fusion=torch.tanh(self.fusion(torch.cat((global_sentence_feature,gt_global_video_feature),dim=1)))

        # batch_size
        f_score=torch.sigmoid(self.linear2(torch.tanh(self.linear1(f_fusion))).squeeze(-1))
        gt_score=torch.sigmoid(self.linear2(torch.tanh(self.linear1(gt_fusion))).squeeze(-1))
        # print(f_score,gt_score)
        # print(torch.nn.BCELoss)
        # batch_size
        loss_aux=-torch.log(f_score)

        # batch_size
        loss_dis=-(torch.log(gt_score)+torch.log(-f_score+1))

        # return loss_aux,loss_dis
        return loss_aux,loss_dis




