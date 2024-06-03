import os

import torch
from torch.utils.data import Dataset

start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]
# * -> değiştirilecek karakterler için
# _ -> değiştirilmeyecek karakterler için
output_vocab = ['*', '_']
diacritic_mapping = {
    'c': 'ç', 'C': 'Ç', 'i': 'ı', 'I': 'İ', 's': 'ş', 'S': 'Ş',
    'o': 'ö', 'O': 'Ö', 'u': 'ü', 'U': 'Ü', 'g': 'ğ', 'G': 'Ğ'
}
chars_that_can_have_diacritics = set(diacritic_mapping.keys())


class Seq2SeqDataset(Dataset):
    def __init__(self, input_sequences, target_sequences, original_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences
        self.original_sequences = original_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx], self.original_sequences[idx]


def prepare_set(input_lines, output_lines):
    if len(input_lines) != len(output_lines):
        raise Exception(f'Input {len(input_lines)} and output lines {len(output_lines)} must have same length')

    result = []
    for input_sentence, ref_sentence in zip(input_lines, output_lines):
        input_sentence = input_sentence.strip()
        ref_sentence = ref_sentence.strip()
        target_sentence = []

        for i, char in enumerate(input_sentence):
            if char in chars_that_can_have_diacritics:
                target_sentence.append('*' if ref_sentence[i] != char else '_')
            else:
                target_sentence.append('_')

        result.append((input_sentence, ''.join(target_sentence), ref_sentence))

    return result


def seq2seq_collate_fn(batch):
    input_seqs, target_seqs, original_seqs = zip(*batch)
    max_input_len = max([len(seq) for seq in input_seqs])
    max_target_len = max([len(seq) for seq in target_seqs])

    padded_inputs = []
    padded_targets = []
    original_sentences = []

    for i in range(len(input_seqs)):
        input_seq = input_seqs[i]
        target_seq = target_seqs[i]
        original_seq = original_seqs[i]

        padded_input = input_seq + [input_vocab.index(end_symbol)] * (max_input_len - len(input_seq))
        padded_target = target_seq + [output_vocab.index('_')] * (max_target_len - len(target_seq))

        padded_inputs.append(padded_input)
        padded_targets.append(padded_target)
        original_sentences.append(original_seq)

    return torch.tensor(padded_inputs, dtype=torch.long), torch.tensor(padded_targets,
                                                                       dtype=torch.long), original_sentences


def prepare_seq2seq_dataset(input_file, target_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        input_sentences = f.readlines()
    with open(target_file, 'r', encoding='utf-8') as f:
        target_sentences = f.readlines()

    data = prepare_set(input_sentences, target_sentences)
    input_sequences = [list(sentence[0]) for sentence in data]
    target_sequences = [list(sentence[1]) for sentence in data]
    original_sequences = [list(sentence[2]) for sentence in data]

    input_indices = [[input_vocab.index(char) for char in seq] for seq in input_sequences]
    target_indices = [[output_vocab.index(char) for char in seq] for seq in target_sequences]

    return Seq2SeqDataset(input_indices, target_indices, original_sequences)
