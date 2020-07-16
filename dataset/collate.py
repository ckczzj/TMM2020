import torch
import torch.nn.functional as F

def collate(batch):
    video=[]
    video_mask=[]
    sentence=[]
    sentence_mask=[]
    gt=[]
    box=[]

    start_gt=[]
    end_gt=[]

    max_video_length=-1
    for i in range(len(batch)):
        if batch[i][0].shape[0]>max_video_length:
            max_video_length=batch[i][0].shape[0]

    max_sentence_length=-1
    for i in range(len(batch)):
        if batch[i][1].shape[0]>max_sentence_length:
            max_sentence_length=batch[i][1].shape[0]

    for i in range(len(batch)):
        feats,words_vec,label=batch[i]
        video_length=feats.shape[0]
        sentence_length=words_vec.shape[0]

        feats=torch.tensor(feats,dtype=torch.float32)
        video.append(F.pad(feats,[0,0,0,max_video_length-video_length]))
        video_mask.append(F.pad(torch.ones(video_length),[0,max_video_length-video_length]))

        words_vec=torch.tensor(words_vec,dtype=torch.float32)
        sentence.append(F.pad(words_vec,[0,0,0,max_sentence_length-sentence_length]))
        sentence_mask.append(F.pad(torch.ones(sentence_length),[0,max_sentence_length-sentence_length]))
        l=torch.zeros(max_video_length)
        l[label[0]:label[1]+1]=1
        gt.append(l)
        box.append(torch.tensor(label))

        l_start=torch.zeros(max_video_length)
        l_start[label[0]]=1
        start_gt.append(l_start)

        l_end=torch.zeros(max_video_length)
        l_end[label[1]]=1
        end_gt.append(l_end)

    video=torch.stack(video,dim=0)
    video_mask=torch.stack(video_mask,dim=0)
    sentence=torch.stack(sentence,dim=0)
    sentence_mask=torch.stack(sentence_mask,dim=0)
    gt=torch.stack(gt,dim=0)
    box=torch.stack(box,dim=0)
    start_gt=torch.stack(start_gt)
    end_gt=torch.stack(end_gt)

    # print(video.shape)
    # print(video_mask.shape)
    # print(sentence.shape)
    # print(sentence_mask.shape)
    # print(gt.shape)
    # print(box)
    # print(box.shape)
    # print("fuck")

    return video,video_mask,sentence,sentence_mask,gt,box,start_gt,end_gt