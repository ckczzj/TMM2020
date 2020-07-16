import torch.nn as nn

from module import BiLSTM
from module import Coordinate
from module import Interaction

class Primary(nn.Module):
    # def __init__(self,args):
    #     super().__init__()
    #
    #     self.video_feature_encoder=BiLSTM(input_size=args.frame_dim,hidden_size=args.hidden_dim)
    #     self.sentence_feature_encoder=BiLSTM(input_size=args.word_dim,hidden_size=args.hidden_dim)
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.video_feature_encoder=BiLSTM(input_size=self.config["frame_dim"],hidden_size=self.config["hidden_dim"])
        self.sentence_feature_encoder=BiLSTM(input_size=self.config["word_dim"],hidden_size=self.config["hidden_dim"])
        # self.interaction=BiInteraction(d=2*self.config["hidden_dim,dropout=self.config["dropout)
        # self.fusion=MultiModalFusion(d=4*self.config["hidden_dim,output_dim=self.config["hidden_dim,dropout=self.config["dropout)
        # self.coordinate=Coordinate(input_size=self.config["hidden_dim,hidden_size=self.config["hidden_dim)

        self.interaction=Interaction(d=2*self.config["hidden_dim"],dropout=self.config["dropout"])
        self.coordinate=Coordinate(input_size=6*self.config["hidden_dim"],hidden_size=self.config["hidden_dim"])
        self.criterion = nn.BCELoss(reduction='none')

    # video: batch_size * video_len * frame_dim
    # video_mask: batch_size * video_len
    # sentence: batch_size * sentence_len * word_dim
    # sentence_mask: batch_size * sentence_len
    # gt: batch_size * video_len
    def forward(self,video,video_mask,sentence,sentence_mask,gt,start_gt,end_gt):
        video_len=video_mask.sum(dim=1)
        sentence_len=sentence_mask.sum(dim=1)

        # video_feature: batch_size * video_len * (2 * hidden_dim)
        video_feature=self.video_feature_encoder(video,video_len)
        # sentence_feature: batch_size * sentence_len * (2 * hidden_dim)
        sentence_feature=self.sentence_feature_encoder(sentence,sentence_len)

        # # video_interaction: batch_size * video_len * (4 * hidden_dim)
        # # sentence_interaction: batch_size * sentence_len * (4 * hidden_dim)
        # video_interaction,sentence_interaction=self.interaction(video_feature,video_mask,sentence_feature,sentence_mask)
        #
        # # batch_size * video_len * (4 * hidden_dim)
        # fusion=self.fusion(video_interaction,video_mask,sentence_interaction,sentence_mask)

        fusion=self.interaction(video_feature, video_mask, sentence_feature, sentence_mask)

        # batch_size * video_len
        # p=self.coordinate(fusion,video_mask)
        # start,end: batch_size
        p,start_score,end_score,start,end=self.coordinate(fusion,video_mask)

        # batch_size * video_len
        start_loss=self.criterion(start_score,start_gt)
        end_loss=self.criterion(end_score,end_gt)

        loss_se=start_loss+end_loss

        # batch_size
        loss_se=loss_se.sum(dim=1)/video_len


        # batch_size * video_len
        loss_loc=self.criterion(p,gt)

        # batch_size
        loss_loc=loss_loc.sum(dim=1)/video_len

        # print(loss_se.mean(),loss_loc.mean())

        return video_feature.clone().detach(),sentence_feature.clone().detach(),p,loss_se,loss_loc,start,end,start_score,end_score

