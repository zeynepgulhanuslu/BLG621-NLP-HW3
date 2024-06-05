import os
from itertools import repeat

diacritic_mapping = {
    'c': 'ç', 'C': 'Ç', 'i': 'ı', 'I': 'İ', 's': 'ş', 'S': 'Ş',
    'o': 'ö', 'O': 'Ö', 'u': 'ü', 'U': 'Ü', 'g': 'ğ', 'G': 'Ğ'
}
start_symbol = '<S>'
end_symbol = '</S>'
input_vocab = [start_symbol] + list(
    'abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]') + [end_symbol]

chars_that_can_have_diacritics = set(diacritic_mapping.keys())


# Veriyi hazırlamak için fonksiyon
def prepare_set(input_lines: list[str], output_lines: list[str], context_size: int) -> list[dict]:
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

            label = "*" if ref[i] != c else "_"
            result.append({'input': input_symbols, 'label': label, 'ref': ref[i]})
    return result


def prepare_binary_dataset(input_file, target_file, context_size=3):
    f_i = open(input_file, 'r', encoding='utf-8')
    f_t = open(target_file, 'r', encoding='utf-8')
    input_sentences = f_i.readlines()
    target_sentences = f_t.readlines()
    return prepare_set(input_sentences, target_sentences, context_size)


result = prepare_set(['kisa siir oku', 'sevdigim uzakta'],
                     ['kısa şiir oku', 'sevdiğim uzakta'],
                     3)

for r in result:
    print(f'{r["input"]} -> {r["label"]} -> {r["ref"]}')