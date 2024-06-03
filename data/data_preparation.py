import os.path
import random

random.seed(123)


def deacritic_conversion(text):
    replace_char_dict = {
        'ı': 'i',
        'İ': 'I',
        'Ü': 'U',
        'ü': 'u',
        'Ş': 'S',
        'ş': 's',
        'Ç': 'C',
        'ç': 'c',
        'Ö': 'O',
        'ö': 'o',
        'Ğ': 'G',
        'ğ': 'g'
    }
    for original_char, replacement_char in replace_char_dict.items():
        text = text.replace(original_char, replacement_char)
    return text


def random_deacritic_conversion(text, probability=0.5):
    replace_char_dict = {
        'ı': 'i',
        'İ': 'I',
        'Ü': 'U',
        'ü': 'u',
        'Ş': 'S',
        'ş': 's',
        'Ç': 'C',
        'ç': 'c',
        'Ö': 'O',
        'ö': 'o',
        'Ğ': 'G',
        'ğ': 'g'
    }
    text_list = list(text)
    for i, char in enumerate(text_list):
        if char in replace_char_dict and random.random() < probability:
            text_list[i] = replace_char_dict[char]
    return ''.join(text_list)


def generate_diacritics_file_with_random(source_file, input_file, target_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Satırları karıştırarak karışık sırada işlem yapalım
    random.shuffle(lines)
    norm_count = 0
    random_count = 0
    f_i = open(input_file, 'w', encoding='utf-8')
    with open(target_file, 'w', encoding='utf-8') as f_o:
        for i, line in enumerate(lines):
            if i < len(lines) / 2:
                # İlk yarı tamamen diacritics değişimi
                norm_count += 1
                d_line = deacritic_conversion(line)
                f_i.write(d_line)
                f_o.write(line)
            else:
                random_count += 1
                # İkinci yarı yüzde elli oranında diacritics değişimi
                d_line = random_deacritic_conversion(line)
                f_i.write(d_line)
                f_o.write(line)
    print(f'random count:{random_count}')
    print(f'norm count:{norm_count}')
    print(f'total lines :{len(open(input_file, "r", encoding="utf-8").readlines())}')


def generate_diacritics_file(source_file, no_diacritics_file, ref_file):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f_i = open(no_diacritics_file, 'w', encoding='utf-8')
    with open(ref_file, 'w', encoding='utf-8') as f_o:
        for i, line in enumerate(lines):
            d_line = deacritic_conversion(line)
            f_i.write(d_line)
            f_o.write(line)
    print(f'total lines :{len(open(no_diacritics_file, "r", encoding="utf-8").readlines())}')


def generate_diacritics_file_both(source_file, no_diacritics_file, ref_file, prob=0.5):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    f_i = open(no_diacritics_file, 'w', encoding='utf-8')

    with open(ref_file, 'w', encoding='utf-8') as f_o:
        for i, line in enumerate(lines):
            d_line = deacritic_conversion(line)
            rand_diacritic_line = random_deacritic_conversion(line, prob)
            # no diacritics file contain ref line, all conversion line and random conversion line.
            f_i.write(line)
            f_i.write(d_line)
            f_i.write(rand_diacritic_line)
            # ref lines for each lines
            f_o.write(line)
            f_o.write(line)
            f_o.write(line)
    print(f'total lines :{len(open(no_diacritics_file, "r", encoding="utf-8").readlines())}')


def process_file(input_file, sub_input_file, sub_target_file, limit):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    subset_lines = lines[:limit]
    half_limit = len(subset_lines) // 2
    norm_count = 0
    random_count = 0
    with open(sub_target_file, 'w', encoding='utf-8') as f_target, \
            open(sub_input_file, 'w', encoding='utf-8') as f_i:
        for i, line in enumerate(subset_lines):
            f_target.write(line)
            if i < half_limit:
                f_i.write(deacritic_conversion(line))
                norm_count += 1
            else:
                f_i.write(random_deacritic_conversion(line))
                random_count += 1

    print(f'total lines:{len(subset_lines)}')
    print(f'random count:{random_count}')
    print(f'norm count:{norm_count}')


def get_subset_data(source_dir, out_dir, train_limit, test_limit):
    os.makedirs(out_dir, exist_ok=True)
    train_norm_file = os.path.join(source_dir, 'train-norm.txt')
    test_norm_file = os.path.join(source_dir, 'testkumesi-gs-norm.txt')

    sub_input_train_f = os.path.join(out_dir, 'train-input.txt')
    sub_input_test_f = os.path.join(out_dir, 'test-input.txt')

    sub_target_train_f = os.path.join(out_dir, 'train-target.txt')
    sub_target_test_f = os.path.join(out_dir, 'test-target.txt')

    # Process train and test files
    process_file(train_norm_file, sub_input_train_f, sub_target_train_f, train_limit)
    process_file(test_norm_file, sub_input_test_f, sub_target_test_f, test_limit)


if __name__ == '__main__':
    source_dir = '../source/original-data'

    out_dir = '../source/all-train-data-v2'
    os.makedirs(out_dir, exist_ok=True)
    train_norm_file = os.path.join(source_dir, 'train-norm.txt')
    train_input_file = os.path.join(out_dir, 'train-input.txt')
    train_target_file = os.path.join(out_dir, 'train-target.txt')

    test_norm_file = os.path.join(source_dir, 'testkumesi-gs-norm.txt')
    test_input_file = os.path.join(out_dir, 'test-input.txt')
    test_target_file = os.path.join(out_dir, 'test-target.txt')

    generate_diacritics_file(train_norm_file, train_input_file, train_target_file)
    generate_diacritics_file(test_norm_file, test_input_file, test_target_file)

    '''
    out_dir = '../source/subset-5k'
    get_subset_data(source_dir, out_dir, 5000, 500)
    '''
