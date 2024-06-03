import argparse
import math
import os
import random
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from data.dnn_data import is_ambiguous
from data.seq2seq_binary_data import prepare_seq2seq_dataset, seq2seq_collate_fn
from Levenshtein import distance as levenshtein_distance

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
chars_that_can_have_diacritics = set(diacritic_mapping.keys())


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input = trg[0, :]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if random.random() < teacher_forcing_ratio else top1

        return outputs


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg, _) in enumerate(iterator):
        src, trg = src.to(model.device), trg.to(model.device)

        optimizer.zero_grad()
        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg, _) in enumerate(iterator):
            src, trg = src.to(model.device), trg.to(model.device)
            output = model(src, trg, 0)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(train_iterator, valid_iterator, input_vocab, output_vocab, model_file):
    INPUT_DIM = len(input_vocab)
    OUTPUT_DIM = len(output_vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = output_vocab.index('_')
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    N_EPOCHS = 10
    CLIP = 1

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_file)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    torch.save(model.state_dict(), model_file)


def inference(model, sentence, input_vocab, output_vocab, max_len=50):
    model.eval()
    tokens = list(sentence)
    src_indexes = [input_vocab.index(token) for token in tokens]
    src_tensor = torch.tensor(src_indexes).unsqueeze(1).to(model.device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [output_vocab.index('*')]

    for i in range(max_len):
        trg_tensor = torch.tensor([trg_indexes[-1]]).to(model.device)
        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        if pred_token == output_vocab.index('_'):
            break

    trg_tokens = [output_vocab[i] for i in trg_indexes]
    return ''.join(trg_tokens[1:])


def get_seq2seq_accuracy(model, test_input_f, test_target_f):
    with open(test_input_f, 'r', encoding='utf-8') as f:
        test_input_sentences = f.readlines()
    with open(test_target_f, 'r', encoding='utf-8') as f:
        test_target_sentences = f.readlines()

    total_amb = 0
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
        transformed_sentence = inference(model, input_sentence, input_vocab, output_vocab).strip()
        lev_dist = levenshtein_distance(target_sentence, transformed_sentence)
        total_distance += lev_dist
        total_chars += len(target_sentence)
        print(f'input: {input_sentence}')
        print(f'target: {target_sentence}')
        print(f'decoded: {transformed_sentence}')
        print('---------------------')
        results.append((input_sentence, target_sentence, transformed_sentence))

        input_words = input_sentence.split()
        target_words = target_sentence.split()
        decoded_words = transformed_sentence.split()

        for input_word, target_word, decoded_word in zip(input_words, target_words, decoded_words):
            if is_ambiguous(target_word):
                total_amb += 1
                total_amb_words += 1
            if target_word == decoded_word:
                if is_ambiguous(target_word):
                    total_correct_amb += 1
                total_correct_words += 1
            total_words += 1

    accuracy = 1 - total_distance / total_chars
    word_accuracy = total_correct_words / total_words * 100
    amb_accuracy = total_correct_amb / total_amb * 100
    print(f'Character-level accuracy: %{accuracy * 100:.2f}')
    print(f'Word-level accuracy: %{word_accuracy:.2f}')
    print(f'Ambiguous accuracy: %{amb_accuracy:.2f}')
    return results, accuracy, word_accuracy, amb_accuracy


def write_results_to_file(results, accuracy, word_accuracy, amb_accuracy, output_file):
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
    parser = argparse.ArgumentParser(description="Train Seq2Seq Model for Diacritics Restoration")
    parser.add_argument("-data_dir", '-d', type=str, required=True, help="Data root for training and test files.")
    parser.add_argument('-model_file', '-m', type=str, required=True, help='Model out file')
    parser.add_argument("-epochs", '-e', type=int, default=10, help="Number of epochs")
    parser.add_argument("-batch_size", '-b', type=int, default=32, help="Batch size")
    parser.add_argument('-result_file', '-r', type=str, required=False, help="Result file for test set results")

    args = parser.parse_args()

    source_dir = args.data_dir
    num_epochs = args.epochs
    batch_size = args.batch_size
    model_file = args.model_file
    result_file = args.result_file

    train_input_f = os.path.join(source_dir, 'train-input.txt')
    train_target_f = os.path.join(source_dir, 'train-target.txt')
    test_input_f = os.path.join(source_dir, 'test-input.txt')
    test_target_f = os.path.join(source_dir, 'test-target.txt')

    train_dataset = prepare_seq2seq_dataset(train_input_f, train_target_f)

    # Eğitim ve doğrulama setlerine bölme
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    test_dataset = prepare_seq2seq_dataset(test_input_f, test_target_f)

    train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=seq2seq_collate_fn)
    valid_iterator = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq2seq_collate_fn)
    test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=seq2seq_collate_fn)

    train_model(train_iterator, valid_iterator, input_vocab, output_vocab, model_file)

    # Modeli yükle
    model = Seq2Seq(
        Encoder(len(input_vocab), 256, 512, 2, 0.5),
        Decoder(len(output_vocab), 256, 512, 2, 0.5),
        device
    ).to(device)
    model.load_state_dict(torch.load(model_file))

    # Test seti doğruluğunu hesapla
    results, accuracy, word_accuracy, amb_accuracy = get_seq2seq_accuracy(model, test_input_f, test_target_f)
    if result_file is not None:
        write_results_to_file(results, accuracy, word_accuracy, amb_accuracy, result_file)
