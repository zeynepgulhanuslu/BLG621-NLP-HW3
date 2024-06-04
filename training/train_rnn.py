import os

import torch
from torch.utils.data import Dataset, DataLoader

from data.seq2seq_binary_data import prepare_seq2seq_dataset

import torch.nn as nn
import torch.nn.functional as F

start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]
# * -> değiştirilecek karakterler için
# _ -> değiştirilmeyecek karakterler için
output_vocab = ['*', '_']


def sequence_to_tensor(sequence, vocab_size):
    tensor = torch.zeros(len(sequence), vocab_size)
    for i, index in enumerate(sequence):
        tensor[i][index] = 1
    return tensor


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


if __name__ == '__main__':
    source_dir = '../source/all-train-data-v1'
    train_input_f = os.path.join(source_dir, 'train-input.txt')
    train_target_f = os.path.join(source_dir, 'train-target.txt')
    test_input_f = os.path.join(source_dir, 'test-input.txt')
    test_target_f = os.path.join(source_dir, 'test-target.txt')

    train_dataset = prepare_seq2seq_dataset(train_input_f, train_target_f)
    n_hidden = 128
    rnn = RNN(len(input_vocab), n_hidden, len(output_vocab))

    input_sequence = train_dataset.input_sequences[0]
    target_sequence = train_dataset.target_sequences[0]
    hidden = torch.zeros(1, n_hidden)

    input_tensor = sequence_to_tensor(input_sequence, len(input_vocab))
    target_tensor = torch.LongTensor(target_sequence)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Örneğin batch_size 32 yaptım.

    
