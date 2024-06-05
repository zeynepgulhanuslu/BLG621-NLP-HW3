import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Levenshtein import distance as levenshtein_distance

from data.dnn_data import is_ambiguous
from data.seq2seq_binary_data import prepare_seq2seq_dataset, seq2seq_collate_fn

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


class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, targets=None):
        embedded = self.embedding(x)
        encoder_outputs, (hidden, cell) = self.encoder(embedded)

        decoder_input = torch.zeros(x.size(0), 1, dtype=torch.long, device=x.device)
        outputs = []

        for t in range(x.size(1)):
            decoder_input_embedded = self.embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder(decoder_input_embedded, (hidden, cell))
            output = self.fc(decoder_output)
            output = self.softmax(output)
            outputs.append(output)

            if targets is not None:
                decoder_input = targets[:, t].unsqueeze(1)  # Teacher forcing
            else:
                decoder_input = output.argmax(2)

        outputs = torch.cat(outputs, dim=1)
        return outputs


def train_seq2seq_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, targets)
            outputs = outputs.view(-1, outputs.shape[-1])
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def seq2seq_infer(model, sentence):
    model.eval()

    input_indices = [input_vocab.index(char) for char in sentence]
    max_input_len = max([len(seq) for seq in sentence])
    padded_input = input_indices + [input_vocab.index(end_symbol)] * (max_input_len - len(input_indices))

    # Padding ekle

    input_tensor = torch.LongTensor(padded_input).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.embedding(input_tensor)
        _, (encoder_hidden, encoder_cell) = model.encoder(embedding)

        # Decoder için hidden ve cell durumlarını sıfırla
        hidden = encoder_hidden
        cell = encoder_cell
        outputs = []
        for t in range(len(sentence)):
            decoder_input = torch.zeros(1, 1, dtype=torch.long, device=device)
            decoder_input_embedded = model.embedding(decoder_input)
            decoder_output, (hidden, cell) = model.decoder(decoder_input_embedded, (hidden, cell))
            output = model.fc(decoder_output)
            output = model.softmax(output)
            topv, topi = output.topk(1)
            # decoder_input = topi.squeeze().detach()

            if topi.item() == input_vocab.index(end_symbol):
                break

            outputs.append(topi.item())

        predicted_labels = [output_vocab[idx] for idx in outputs]

    return ''.join(predicted_labels)


def apply_diacritics(sentence, labels):
    transformed_sentence = []
    for char, label in zip(sentence, labels):
        if label == '*':
            transformed_char = diacritic_mapping.get(char, char)
        else:
            transformed_char = char
        transformed_sentence.append(transformed_char)
    return ''.join(transformed_sentence)


def get_seq2seq_test_accuracy(model, test_input_f, test_target_f):
    f_i = open(test_input_f, 'r', encoding='utf-8')
    f_t = open(test_target_f, 'r', encoding='utf-8')
    test_input_sentences = f_i.readlines()
    test_target_sentences = f_t.readlines()

    total_correct_amb = 0
    total_distance = 0
    total_chars = 0
    total_correct_words = 0
    total_words = 0
    total_amb_words = 0
    results = []
    for input_sentence, target_sentence in zip(test_input_sentences, test_target_sentences):
        input_sentence = input_sentence.strip()
        target_sentence = target_sentence.strip()
        transformed_sentence = seq2seq_infer(model, input_sentence).strip()
        transformed_sentence_norm = apply_diacritics(input_sentence, transformed_sentence)
        lev_dist = levenshtein_distance(target_sentence, transformed_sentence_norm)
        total_distance += lev_dist
        total_chars += len(target_sentence)
        print(f'input: {input_sentence}')
        print(f'target: {target_sentence}')
        print(f'decoded: {transformed_sentence_norm.strip()}')
        print('---------------------')
        results.append((input_sentence, target_sentence, transformed_sentence_norm))

        input_words = input_sentence.split()
        target_words = target_sentence.split()
        decoded_words = transformed_sentence_norm.split()

        for input_word, target_word, decoded_word in zip(input_words, target_words, decoded_words):
            if is_ambiguous(target_word):
                total_amb_words += 1
            if target_word == decoded_word:
                if is_ambiguous(target_word):
                    total_correct_amb += 1
                total_correct_words += 1
            total_words += 1

    accuracy = 1 - total_distance / total_chars
    word_accuracy = total_correct_words / total_words * 100
    amb_accuracy = total_correct_amb / total_amb_words * 100
    print(f'Character-level accuracy: %{accuracy * 100:.2f}')
    print(f'Word-level accuracy: %{word_accuracy:.2f}')
    print(f'Ambiguous accuracy: %{amb_accuracy:.2f}')
    return results, accuracy, word_accuracy, amb_accuracy


def write_results_to_file(results: list, accuracy: float, word_accuracy: float, amb_accuracy: float, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for input_sentence, target_sentence, transformed_sentence in results:
            f.write(f"Input: {input_sentence}\n")
            f.write(f"Target: {target_sentence}\n")
            f.write(f"Output: {transformed_sentence}\n")
            f.write('---------------------\n')
        f.write(f'Character-level accuracy: %{accuracy * 100:.2f}\n')
        f.write(f'Word-level accuracy: %{word_accuracy:.2f}\n')
        f.write(f'Ambiguous accuracy: %{amb_accuracy:.2f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DNN Model for Diacritics Restoration")
    parser.add_argument("-data_dir", '-d', type=str, required=True,
                        help="Data root for training and test files.")
    parser.add_argument("-embed_size", '-emb', type=int, default=3,
                        help="Context size for character concat")
    parser.add_argument("-hidden_dim", '-dim', type=int, default=1024,
                        help="Hidden dimension for model")
    parser.add_argument("-num_layers", '-l', type=int, default=3,
                        help="Number of layers")
    parser.add_argument("-epochs", '-e', type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("-batch_size", '-b', type=int, default=32,
                        help="Batch size")
    parser.add_argument("-model_file", '-m', type=str, required=True,
                        help="Model out file.")
    parser.add_argument("-result_file", '-r', type=str, required=False,
                        help="Result file for test set results")

    args = parser.parse_args()

    source_dir = args.data_dir
    embed_size = args.embed_size
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    num_epochs = args.epochs
    batch_size = args.batch_size
    model_file = args.model_file
    result_file = args.result_file

    train_input_f = os.path.join(source_dir, 'train-input.txt')
    train_target_f = os.path.join(source_dir, 'train-target.txt')
    test_input_f = os.path.join(source_dir, 'test-input.txt')
    test_target_f = os.path.join(source_dir, 'test-target.txt')

    dataset = prepare_seq2seq_dataset(train_input_f, train_target_f)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=seq2seq_collate_fn)

    input_vocab_size = len(input_vocab)
    output_vocab_size = len(output_vocab)
    dropout = 0.5

    model = Seq2SeqLSTM(input_vocab_size, output_vocab_size, embed_size,
                        hidden_dim, num_layers, dropout).to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_seq2seq_model(model, dataloader, criterion, optimizer, num_epochs, device)
    torch.save(model.state_dict(), model_file)

    model.eval()
