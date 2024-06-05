import os

import torch

from training.train_dnn_model_binary import FeedForwardNN, get_dnn_test_accuracy, write_results_to_file

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


model_file = '../source/models/dnn/binary-dnn-v2-dim=1024c=5layer=3.pth'
context_size = 5
input_dim = len(input_vocab) * (2 * context_size + 1)  # Adjusted for the context window
output_dim = len(output_vocab)
dropout_prob = 0.5  #

model = FeedForwardNN(input_dim, 1024, output_dim, 3, dropout_prob)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()
print(model)
source_dir = '../source/all-train-data-v2'
test_input_f = os.path.join(source_dir, 'test-input.txt')
test_target_f = os.path.join(source_dir, 'test-target.txt')
result_file = '../source/results/binary-dnn-v2-dim=1024c=5layer=3.txt'
results, accuracy, word_accuracy, amb_accuracy = get_dnn_test_accuracy(model, test_input_f, test_target_f,
                                                                       context_size)
if result_file is not None:
    write_results_to_file(results, accuracy, word_accuracy, amb_accuracy, result_file)
