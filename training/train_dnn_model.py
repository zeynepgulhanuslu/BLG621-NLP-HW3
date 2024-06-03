import argparse
import os
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Levenshtein import distance as levenshtein_distance
from torch.utils.data import Dataset, DataLoader

from data.dnn_data import prepare_dataset_with_pairs, is_ambiguous

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chars_that_can_have_diacritics = set('cCiIsSoOuUgG')

output_vocab = list('cCçÇıIiİsSşŞoOöÖuUüÜgGğĞ')
start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]
input_vocab_dict = {c: i for i, c in enumerate(input_vocab)}
output_vocab_dict = {i: c for i, c in enumerate(output_vocab)}


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout_prob)
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            x = self.dropout(x)
        output = self.output_layer(x)
        return output


class DiacriticsDataset(Dataset):
    def __init__(self, training_elements, input_vocab, output_vocab):
        self.training_elements = training_elements
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_vocab_size = len(self.input_vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.input_vocab_dict = {c: i for i, c in enumerate(self.input_vocab)}
        self.output_vocab_dict = {c: i for i, c in enumerate(self.output_vocab)}

    def __len__(self):
        return len(self.training_elements)

    def __getitem__(self, idx):
        input_symbols = self.training_elements[idx].input
        input_indices = [self.input_vocab_dict[symbol] for symbol in input_symbols]
        target_char = self.training_elements[idx].ref
        target_index = self.output_vocab_dict[target_char]

        input_one_hot = np.zeros((len(input_symbols), self.input_vocab_size), dtype=np.float32)
        for i, idx in enumerate(input_indices):
            input_one_hot[i, idx] = 1.0

        return torch.tensor(input_one_hot, dtype=torch.float32), torch.tensor(target_index, dtype=torch.long)


def train_dnn_model(model, dataloader, criterion, optimizer, num_epochs, model_file):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.view(inputs.size(0), -1)  # Reshape inputs to match expected dimensions
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')
    torch.save(model.state_dict(), model_file)


def get_vector_as_tensor(i, sentence, context_size):
    input_symbols: list[str] = []
    start = i - context_size
    end = i + context_size + 1
    if start < 0:
        input_symbols.extend(repeat(start_symbol, context_size - i))
        input_symbols.extend(sentence[0:i])
    else:
        input_symbols.extend(sentence[start:i])
    if end > len(sentence):
        input_symbols.extend(sentence[i:len(sentence)])
        input_symbols.extend(repeat(end_symbol, end - len(sentence)))
    else:
        input_symbols.extend(sentence[i:end])
    input_indices = [input_vocab_dict[symbol] for symbol in input_symbols]
    input_one_hot = np.zeros((len(input_symbols), len(input_vocab)), dtype=np.float32)
    for i, idx in enumerate(input_indices):
        input_one_hot[i, idx] = 1.0

    input_one_hot = torch.tensor(input_one_hot, dtype=torch.float32)

    input_tensor = torch.tensor(input_one_hot, dtype=torch.float32).unsqueeze(0).to(device)
    input_tensor = input_tensor.view(input_tensor.size(0), -1)  # Reshape and convert to float

    return input_tensor


def dnn_infer(model, input_sentence, context_size):
    model.eval()
    input_sentence = list(input_sentence)
    transformed_sentence = []

    with torch.no_grad():
        for i, char in enumerate(input_sentence):
            if char in chars_that_can_have_diacritics:
                input_tensor = get_vector_as_tensor(i, input_sentence, context_size)

                output = model(input_tensor)

                _, predicted_idx = torch.max(output, 1)
                transformed_char = output_vocab[predicted_idx.item()]
                transformed_sentence.append(transformed_char)
            else:
                transformed_sentence.append(char)

    transformed_sentence = ''.join(transformed_sentence)
    return transformed_sentence


def get_dnn_test_accuracy(model, test_input_f, test_target_f):
    f_i = open(test_input_f, 'r', encoding='utf-8')
    f_t = open(test_target_f, 'r', encoding='utf-8')
    test_input_sentences = f_i.readlines()
    test_target_sentences = f_t.readlines()

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
        transformed_sentence = dnn_infer(model, input_sentence, context_size).strip()
        lev_dist = levenshtein_distance(target_sentence, transformed_sentence)
        total_distance += lev_dist
        total_chars += len(target_sentence)
        print(f'input: {input_sentence}')
        print(f'target: {target_sentence}')
        print(f'decoded: {transformed_sentence.strip()}')
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
    parser.add_argument("-context_size", '-c', type=int, default=3,
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
    context_size = args.context_size
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

    training_elements = prepare_dataset_with_pairs(train_input_f, train_target_f, context_size)
    train_dataset = DiacriticsDataset(training_elements, input_vocab, output_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(input_vocab) * (2 * context_size + 1)  # Adjusted for the context window
    output_dim = len(output_vocab)
    dropout_prob = 0.5  #

    model = FeedForwardNN(input_dim, hidden_dim, output_dim, num_layers, dropout_prob)
    print(f'model initialized with \ninput dim: {input_dim}\nout_dim:{output_dim}\n'
          f'hidden dim:{hidden_dim}\nnum layers: {num_layers}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model = model.to(device)

    train_dnn_model(model, train_dataloader, criterion, optimizer, num_epochs, model_file)
    # get test result

    results, accuracy, word_accuracy, amb_accuracy = get_dnn_test_accuracy(model, test_input_f, test_target_f)
    if result_file is not None:
        write_results_to_file(results, accuracy, word_accuracy, amb_accuracy, result_file)
