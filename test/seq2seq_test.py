import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.seq2seq_binary_data import prepare_seq2seq_dataset, seq2seq_collate_fn
from training.train_seq2seq import get_seq2seq_test_accuracy, Seq2SeqLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diacritic_mapping = {
    'c': 'ç', 'C': 'Ç', 'i': 'ı', 'I': 'İ', 's': 'ş', 'S': 'Ş',
    'o': 'ö', 'O': 'Ö', 'u': 'ü', 'U': 'Ü', 'g': 'ğ', 'G': 'Ğ'
}
start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]
# * -> değiştirilecek karakterler için
# _ -> değiştirilmeyecek karakterler için
output_vocab = ['*', '_']
chars_that_can_have_diacritics = set('cCiIsSoOuUgG')

if __name__ == '__main__':
    input_vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)
    dropout = 0.5
    embed_size = 128
    hidden_dim = 256
    num_layers = 3
    model = Seq2SeqLSTM(input_vocab_size, output_vocab_size, embed_size,
                        hidden_dim, num_layers, dropout).to(device)
    model_file = '../source/models/seq2seq/seq2seq-v1.pth'
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    print(model)
    source_dir = '../source/all-train-data-v1'
    test_input_f = os.path.join(source_dir, 'test-input.txt')
    test_target_f = os.path.join(source_dir, 'test-target.txt')

    results, accuracy, word_accuracy, amb_accuracy = get_seq2seq_test_accuracy(model, test_input_f, test_target_f)
