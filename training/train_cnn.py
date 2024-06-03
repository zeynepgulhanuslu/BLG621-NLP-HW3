import argparse
import os

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from data.dnn_binary_data import prepare_binary_dataset
from data.dnn_data import is_ambiguous
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
output_vocab = ['*', '_']
chars_that_can_have_diacritics = set(diacritic_mapping.keys())


class DiacriticsBinaryDataset(Dataset):
    def __init__(self, training_elements, input_vocab, output_vocab):
        self.training_elements = training_elements
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.input_vocab_size = len(self.input_vocab)
        self.output_vocab_size = len(self.output_vocab)
        self.input_vocab_dict = {c: i for i, c in enumerate(self.input_vocab)}
        self.output_vocab_dict = {c: i for i, c in enumerate(self.output_vocab)}
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(["*", "_"])

    def __len__(self):
        return len(self.training_elements)

    def __getitem__(self, idx):
        input_symbols = self.training_elements[idx]['input']
        input_indices = [self.input_vocab_dict[symbol] for symbol in input_symbols]
        label = self.training_elements[idx]['label']
        label_index = self.label_encoder.transform([label])[0]

        input_one_hot = np.zeros((len(input_symbols), self.input_vocab_size), dtype=np.float32)
        for i, idx in enumerate(input_indices):
            input_one_hot[i, idx] = 1.0

        return torch.tensor(input_one_hot, dtype=torch.float32), torch.tensor(label_index, dtype=torch.long)


class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.max(x, dim=2)[0]
        x = self.fc1(x)
        return x


# Model ve eğitim fonksiyonları
def train_dnn_model_binary(model, dataloader, criterion, optimizer, num_epochs, model_file):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')
    torch.save(model.state_dict(), model_file)


def get_vector_as_tensor(index, input_sentence, context_size):
    start_padding = [start_symbol] * context_size
    end_padding = [end_symbol] * context_size
    padded_sentence = start_padding + input_sentence + end_padding

    context_window = padded_sentence[index: index + 2 * context_size + 1]
    input_indices = [input_vocab.index(char) for char in context_window]
    input_one_hot = np.zeros((len(context_window), len(input_vocab)), dtype=np.float32)
    for i, idx in enumerate(input_indices):
        input_one_hot[i, idx] = 1.0

    return torch.tensor(input_one_hot, dtype=torch.float32).to(device).view(-1, len(input_vocab))


def dnn_binary_infer(model, input_sentence, context_size):
    model.eval()
    input_sentence = list(input_sentence)
    transformed_sentence = []

    with torch.no_grad():
        for i, char in enumerate(input_sentence):
            if char in chars_that_can_have_diacritics:
                input_tensor = get_vector_as_tensor(i + context_size, input_sentence, context_size)
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                predicted_label = output_vocab[predicted_idx.item()]
                if predicted_label == "*":
                    transformed_char = diacritic_mapping.get(char, char)
                else:
                    transformed_char = char
                transformed_sentence.append(transformed_char)
            else:
                transformed_sentence.append(char)

    transformed_sentence = ''.join(transformed_sentence)
    return transformed_sentence


def get_dnn_test_accuracy(model, test_input_f, test_target_f, context_size):
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
        transformed_sentence = dnn_binary_infer(model, input_sentence, context_size).strip()
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
    parser.add_argument("-hidden_dim", '-dim', type=int, default=128,
                        help="Hidden dimension for model")
    parser.add_argument("-num_layers", '-l', type=int, default=2,
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

    train_pairs = prepare_binary_dataset(train_input_f, train_target_f, context_size)
    train_dataset = DiacriticsBinaryDataset(train_pairs, input_vocab, output_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    dropout_prob = 0.5

    model = CNNModel(input_dim, hidden_dim, output_dim, dropout_prob)
    print(f'model initialized with \ninput dim: {input_dim}\nout_dim:{output_dim}\n'
          f'hidden dim:{hidden_dim}\nnum layers: {num_layers}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    model = model.to(device)

    train_dnn_model_binary(model, train_dataloader, criterion, optimizer, num_epochs, model_file)
    # get test result

    results, accuracy, word_accuracy, amb_accuracy = get_dnn_test_accuracy(model, test_input_f, test_target_f,
                                                                           context_size)
    if result_file is not None:
        write_results_to_file(results, accuracy, word_accuracy, amb_accuracy, result_file)
