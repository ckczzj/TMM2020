import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bias=True, batch_first=True, bidirectional=True)

    # batch: batch_size * len * input_size
    # seq_len: batch_size
    def forward(self, batch, seq_len):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, descending=True)
        _, original_idx = torch.sort(sorted_idx)

        sorted_batch = batch.index_select(0, sorted_idx)

        packed_batch = nn.utils.rnn.pack_padded_sequence(
            sorted_batch, sorted_seq_len.cpu().data.numpy(), batch_first=True)

        out, _ = self.lstm(packed_batch)

        unpacked_batch, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        out = unpacked_batch.index_select(0, original_idx)

        # if out.shape[1] < max_frame_num:
        #     out = F.pad(out, [0, 0, 0, max_frame_num - out.shape[1]])

        # out: batch_size * max_frame_num * (2 * hidden_size)
        return out
