import collections
import logging
import os
import argparse

import torch

from dataset import collate
from torch.utils.data import DataLoader
from dataset import ActivityNet
from dataset import TACoS

import criteria

from model import Primary
from model import Auxiliary
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.inverse_square_root_schedule import InverseSquareRootSchedule
from utils import load_word2vec, AverageMeter, TimeMeter
import numpy as np


class Runner:
    def __init__(self, config):
        self.pre_train_num_updates = 0
        self.dis_num_updates = 0
        self.gen_num_updates = 0
        self.config = config
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config["gpu"]
        self._build_loader()
        self._build_model()
        self._build_optimizer()

    def _get_dataset(self):
        word2vec = load_word2vec(self.config["word2vec_path"])
        if self.config["dataset"] == "ActivityNet":
            train=ActivityNet(
                self.config["feature_path"],
                self.config["train_json"],
                word2vec,
                max_frame_num=self.config["max_frame_num"],
                max_word_num=self.config["max_word_num"]
            )
            test=ActivityNet(
                self.config["feature_path"],
                self.config["test_json"],
                word2vec,
                max_frame_num=self.config["max_frame_num"],
                max_word_num=self.config["max_word_num"]
            )
        elif self.config["dataset"] == "TACoS":
            train=TACoS(
                self.config["feature_path"],
                self.config["train_json"],
                word2vec,
                max_frame_num=self.config["max_frame_num"],
                max_word_num=self.config["max_word_num"]
            )
            test=TACoS(
                self.config["feature_path"],
                self.config["test_json"],
                word2vec,
                max_frame_num=self.config["max_frame_num"],
                max_word_num=self.config["max_word_num"]
            )

        else:
            raise NotImplementedError

        return train,test

    def _build_loader(self):
        train,test = self._get_dataset()

        self.train_loader = DataLoader(dataset=train, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], shuffle=True, collate_fn=collate)
        self.test_loader = DataLoader(dataset=test, batch_size=self.config["batch_size"], num_workers=self.config["num_workers"], shuffle=False, collate_fn=collate)

    def _build_model(self):
        if self.config["use_pre_train_primary"]:
            self.primary=torch.load(os.path.join(self.config["model_save_path"],"pre-train-model.pth")).cuda()
        else:
            self.primary = Primary(self.config).cuda()
        self.auxiliary = Auxiliary(self.config).cuda()

    def _build_optimizer(self):
        pre_train_parser = argparse.ArgumentParser()
        pre_train_parser.add_argument("--lr", type=float, default=self.config["pre_train_lr"], help="")
        pre_train_parser.add_argument("--warmup-updates", type=int, default=self.config["pre_train_warmup_updates"], help="")
        pre_train_parser.add_argument("--warmup-init-lr", type=float, default=self.config["pre_train_warmup_init_lr"], help="")
        FairseqAdam.add_args(pre_train_parser)
        pre_train_args=pre_train_parser.parse_args()
        self.pre_train_optimizer = FairseqAdam(pre_train_args, self.primary.parameters())
        self.pre_train_lr_scheduler = InverseSquareRootSchedule(pre_train_args, self.pre_train_optimizer)
        # self.pre_train_optimizer = Adam(self.primary.parameters(),lr=1e-4)

        generator_parser = argparse.ArgumentParser()
        generator_parser.add_argument("--lr", type=float, default=self.config["generator_lr"], help="")
        generator_parser.add_argument("--warmup-updates", type=int, default=self.config["generator_warmup_updates"], help="")
        generator_parser.add_argument("--warmup-init-lr", type=float, default=self.config["generator_warmup_init_lr"], help="")
        FairseqAdam.add_args(generator_parser)
        generator_args=generator_parser.parse_args()
        self.generator_optimizer = FairseqAdam(generator_args,self.primary.parameters())
        self.generator_lr_scheduler = InverseSquareRootSchedule(generator_args, self.generator_optimizer)


        discriminator_parser = argparse.ArgumentParser()
        discriminator_parser.add_argument("--lr", type=float, default=self.config["discriminator_lr"], help="")
        discriminator_parser.add_argument("--warmup-updates", type=int, default=self.config["discriminator_warmup_updates"], help="")
        discriminator_parser.add_argument("--warmup-init-lr", type=float, default=self.config["discriminator_warmup_init_lr"], help="")
        FairseqAdam.add_args(discriminator_parser)
        discriminator_args=discriminator_parser.parse_args()
        self.discriminator_optimizer = FairseqAdam(discriminator_args,list(self.auxiliary.parameters()))
        self.discriminator_lr_scheduler = InverseSquareRootSchedule(discriminator_args, self.discriminator_optimizer)

    def train(self):
        if not self.config["use_pre_train_primary"]:
            logging.info("Pre-train")
            for epoch in range(1, self.config["pre_train_epoch_num"] + 1):
                logging.info("Start pre-train Epoch {}".format(epoch))
                self._pre_train_one_epoch(epoch)
                self.eval()
                if self.config["save_model"]:
                    torch.save(self.primary,os.path.join(self.config["model_save_path"],"pre-train-model-"+str(epoch)+".pth"))

        else:
            logging.info("pre-train eval")
            self.eval()
            logging.info("adversarial")
            for epoch in range(1, self.config["adversarial_epoch_num"] + 1):
                logging.info("Start Adversarial Epoch {}".format(epoch))
                self._adversarial_one_epoch(epoch)
                if epoch>self.config["pre_train_dis_epoch_num"]:
                    self.eval()

        logging.info("Done.")

    def _adversarial_one_epoch(self,epoch):
        self.primary.train()
        self.auxiliary.train()
        time_meter = TimeMeter()
        loss_dis_meter = AverageMeter()

        for batch_id,(video,video_mask,sentence,sentence_mask,gt,box,start_gt,end_gt) in enumerate(self.train_loader, 1):
            if epoch<=self.config["pre_train_dis_epoch_num"]:
                video_feature,sentence_feature,p,loss_se,loss_loc,_,_,_,_=self.primary(video.cuda(),video_mask.cuda(),sentence.cuda(),sentence_mask.cuda(),gt.cuda(),start_gt.cuda(),end_gt.cuda())
                _,loss_dis=self.auxiliary(video_feature,video_mask.cuda(),sentence_feature,sentence_mask.cuda(),p.detach(),gt.cuda())
                loss_dis=loss_dis.mean()
                self.discriminator_optimizer.zero_grad()
                loss_dis.backward()
                self.discriminator_optimizer.step()
                self.dis_num_updates += 1
                self.discriminator_lr_scheduler.step_update(self.dis_num_updates)
                loss_dis_meter.update(loss_dis.item())

            else:
                video_feature,sentence_feature,p,loss_se,loss_loc,_,_,_,_=self.primary(video.cuda(),video_mask.cuda(),sentence.cuda(),sentence_mask.cuda(),gt.cuda(),start_gt.cuda(),end_gt.cuda())
                _,loss_dis=self.auxiliary(video_feature,video_mask.cuda(),sentence_feature,sentence_mask.cuda(),p,gt.cuda())
                loss_dis=loss_dis.mean()
                self.discriminator_optimizer.zero_grad()
                loss_dis.backward()
                self.discriminator_optimizer.step()
                self.dis_num_updates += 1
                self.discriminator_lr_scheduler.step_update(self.dis_num_updates)

                video_feature,sentence_feature,p,loss_se,loss_loc,_,_,_,_=self.primary(video.cuda(),video_mask.cuda(),sentence.cuda(),sentence_mask.cuda(),gt.cuda(),start_gt.cuda(),end_gt.cuda())
                loss_aux,_=self.auxiliary(video_feature,video_mask.cuda(),sentence_feature,sentence_mask.cuda(),p,gt.cuda())
                loss_gen=self.config["lambda"]*loss_se+self.config["mu"]*loss_loc+self.config["nu"]*loss_aux
                # print(self.config["lamda*loss_se.mean(),self.config["mu*loss_loc.mean(),self.config["nu*loss_aux.mean())
                loss_gen=loss_gen.mean()
                self.generator_optimizer.zero_grad()
                loss_gen.backward()
                self.generator_optimizer.step()
                self.gen_num_updates+=1
                self.generator_lr_scheduler.step_update(self.gen_num_updates)


            time_meter.update()


            if batch_id % self.config["display_n_batches"] == 0:
                if epoch<=self.config["pre_train_dis_epoch_num"]:
                    logging.info("Adversarial Epoch %d, Dis_loss %.4f, Batch %d, %.3f seconds/batch" % (
                        epoch, loss_dis_meter.avg, batch_id, 1.0 / time_meter.avg
                    ))
                    loss_dis_meter.reset()
                else:
                    logging.info("Adversarial Epoch %d, Batch %d, %.3f seconds/batch" % (
                        epoch, batch_id, 1.0 / time_meter.avg
                    ))
                    loss_dis_meter.reset()


    def _pre_train_one_epoch(self, epoch):
        self.primary.train()
        loss_meter = AverageMeter()
        time_meter = TimeMeter()
        for batch_id,(video,video_mask,sentence,sentence_mask,gt,box,start_gt,end_gt) in enumerate(self.train_loader, 1):
            self.pre_train_optimizer.zero_grad()

            _,_,_,loss_se,loss_loc,_,_,_,_ = self.primary(video.cuda(),video_mask.cuda(),sentence.cuda(),sentence_mask.cuda(),gt.cuda(),start_gt.cuda(),end_gt.cuda())

            # batch_size
            # print(self.config["alpha*loss_se.mean(),self.config["beta*loss_loc.mean())
            loss=self.config["alpha"]*loss_se+self.config["beta"]*loss_loc

            loss=loss.mean()

            # loss.backward()
            self.pre_train_optimizer.backward(loss)

            self.pre_train_optimizer.step()
            self.pre_train_num_updates += 1
            curr_lr = self.pre_train_lr_scheduler.step_update(self.pre_train_num_updates)

            loss_meter.update(loss.item())
            time_meter.update()

            if batch_id % self.config["display_n_batches"] == 0:
                logging.info("Pre-train Epoch %d, Batch %d, loss = %.4f, lr = %.6f, %.3f seconds/batch" % (
                    epoch, batch_id, loss_meter.avg, curr_lr, 1.0 / time_meter.avg
                ))
                loss_meter.reset()

    def eval(self):
        data_loaders = [self.test_loader]
        meters = collections.defaultdict(lambda: AverageMeter())

        self.primary.eval()
        with torch.no_grad():
            for data_loader in data_loaders:
                for batch_id,(video,video_mask,sentence,sentence_mask,gt,box,start_gt,end_gt) in enumerate(data_loader, 1):
                    _,_,_, loss_se,loss_loc,start,end,start_score,end_score = self.primary(video.cuda(),video_mask.cuda(),sentence.cuda(),sentence_mask.cuda(),gt.cuda(),start_gt.cuda(),end_gt.cuda())

                    box = box.cpu().numpy()
                    box_start, box_end = box[:, 0], box[:, 1]

                    video_len = np.sum(video_mask.int().cpu().numpy(), -1)
                    predict_start=start.cpu().numpy()
                    predict_end=end.cpu().numpy()
                    tmp_start=predict_start.copy()
                    swap_indices=predict_start>predict_end
                    predict_start[swap_indices]=predict_end[swap_indices]
                    predict_end[swap_indices]=tmp_start[swap_indices]

                    predict_start[predict_start < 0] = 0
                    predict_end[predict_end >= video_len] = video_len[predict_end >= video_len] - 1

                    IoUs = criteria.calculate_IoU_batch((predict_start, predict_end),
                                                        (box_start, box_end))
                    meters["mIoU"].update(np.mean(IoUs), IoUs.shape[0])
                    for i in range(1, 8, 2):
                        meters["IoU@0.%d" % i].update(np.mean(IoUs >= (i / 10)), IoUs.shape[0])
                # all val or test metric
                print("| ")
                for key, value in meters.items():
                    print("{}, {:.4f}".format(key, value.avg), end=" | ")
                    meters[key].reset()
                print("\n| ")