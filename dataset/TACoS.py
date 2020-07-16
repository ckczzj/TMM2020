import os
import numpy as np
from torch.utils.data import Dataset
from utils import load_feature, load_json, tokenize


class TACoS(Dataset):
    def __init__(self, feature_path, data_path, word2vec, max_frame_num, max_word_num):
        self.data = load_json(data_path)
        self.feature_path = feature_path
        self.word2vec = word2vec
        self.max_frame_num = max_frame_num
        self.max_word_num = max_word_num

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        feats = load_feature(os.path.join(self.feature_path, '%s.npy' % vid[:-4]), dataset='TACoS')
        video_len = feats.shape[0]

        fps = video_len / duration
        start_frame = int(fps * timestamps[0])
        end_frame = int(fps * timestamps[1])
        if end_frame >= video_len:
            end_frame = video_len - 1
        if start_frame > end_frame:
            start_frame = end_frame
        assert start_frame <= end_frame
        assert 0 <= start_frame < video_len
        assert 0 <= end_frame < video_len
        label = np.asarray([start_frame, end_frame]).astype(np.int32)

        # down sample
        if video_len > self.max_frame_num:
            index = np.linspace(start=0, stop=video_len - 1, num=self.max_frame_num).astype(np.int32)
            new_feats = []
            for i in range(len(index) - 1):
                start = index[i]
                end = index[i + 1]
                if start == end or start + 1 == end:
                    new_feats.append(feats[start])
                else:
                    new_feats.append(np.mean(feats[start: end], 0))
            new_feats.append(feats[-1])
            feats = np.stack(new_feats, 0)

            assert feats.shape[0]<=self.max_frame_num

            label[0] = min(np.where(index >= label[0])[0])
            if label[1] == video_len - 1:
                label[1] = self.max_frame_num - 1
            else:
                label[1] = max(np.where(index <= label[1])[0])
            if label[1] < label[0]:
                label[0] = label[1]

            assert label[0] <= label[1]
            assert 0 <= label[0] < self.max_frame_num
            assert 0 <= label[1] < self.max_frame_num

        words = tokenize(sentence, self.word2vec)
        words_vec = np.asarray([self.word2vec[word] for word in words])
        words_vec = words_vec.astype(np.float32)
        sentence_len = words_vec.shape[0]
        if sentence_len > self.max_word_num:
            words_vec=words_vec[:self.max_word_num,:]

        return feats, words_vec, label

    def __len__(self):
        return len(self.data)
