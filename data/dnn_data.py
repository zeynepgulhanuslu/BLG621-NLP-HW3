"""
Burada önceki ve sonraki karakterler dikkate alınarak bir veri seti hazırlanmıştır.
Sadece ambigous karakterler için giriş çıkış değerleri belirlenecek şekildedir.
Bu şekilde basit bir dnn modeli eğitilmiştir.
"""
from itertools import repeat


chars_that_can_have_diacritics = set('cCiIsSoOuUgG')

output_vocab = list('cCçÇıIiİsSşŞoOöÖuUüÜgGğĞ')
start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]


def load_lines(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()


class TrainingElement:
    def __init__(self, input: list[str], ref: str):
        self.input = input
        self.ref = ref


def prepare_set(input_lines: list[str], output_lines: list[str], context_size: int) -> list[TrainingElement]:
    if len(input_lines) != len(output_lines):
        raise Exception(f'input {len(input_lines)} and output lines {len(output_lines)}  must have same length')
    pairs = zip(input_lines, output_lines)
    result = []
    for pair in pairs:
        input = pair[0].strip()
        ref = pair[1].strip()
        for i, c in enumerate(input):
            if c not in chars_that_can_have_diacritics:
                continue
            input_symbols: list[str] = []
            start = i - context_size
            end = i + context_size + 1

            if start < 0:
                input_symbols.extend(repeat(start_symbol, context_size - i))
                input_symbols.extend(input[0:i])
            else:
                input_symbols.extend(input[start:i])

            if end > len(input):
                input_symbols.extend(input[i:len(input)])
                input_symbols.extend(repeat(end_symbol, end - len(input)))
            else:
                input_symbols.extend(input[i:end])

            result.append(TrainingElement(input_symbols, ref[i]))
    return result


def prepare_dataset_with_pairs(input_file, target_file, context_size=3):
    f_i = open(input_file, 'r', encoding='utf-8')
    f_t = open(target_file, 'r', encoding='utf-8')
    input_sentences = f_i.readlines()
    target_sentences = f_t.readlines()
    return prepare_set(input_sentences, target_sentences, context_size)


def is_ambiguous(word):
    for c in word:
        if c in chars_that_can_have_diacritics:
            return True
    return False


if __name__ == '__main__':
    context_size = 3
    input_vocab_size = len(input_vocab)
    print(f'Input vocab size = {input_vocab_size}')
    output_vocab_size = len(output_vocab)
    print(f'Output vocab size = {output_vocab_size}')
    print(f'input dimension = {input_vocab_size * (2 * context_size + 1)}')

    result = prepare_set(['kisa siir oku', 'oldugun gıbı gorun'],
                         ['kısa şiir oku', 'olduğun gibi görün'],
                         3)
    print(f'type: {type(result)}')
    for r in result:
        print(f'{r.input} -> {r.ref}')
        print('-----------------')
