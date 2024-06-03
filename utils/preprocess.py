import string
import re

# Türkçe karakterler ve boşluk
turkish_characters = "A-Za-zÇçĞğİıÖöŞşÜü "

# Düzenli ifade: Türkçe karakterler ve boşluk dışındaki karakterleri bulmak için
pattern = f"[^{turkish_characters}]"

MAX_SENTENCE_LEN = 100


def get_non_alphabetic_characters(file_path):
    # Alfabe harflerini tanımlayın (küçük ve büyük harfler)
    alphabetic_characters = set(string.ascii_letters)

    # Benzersiz olmayan alfabe dışı karakterleri toplamak için bir set oluşturun
    non_alphabetic_characters = set()

    # Dosyayı okuyun
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                for character in line:
                    if character not in alphabetic_characters:
                        non_alphabetic_characters.add(character)

    return non_alphabetic_characters


def clean_file(input_file, output_file, removed_file):
    pattern = re.compile(
        r'[abcçdefgğhıijklmnoöprsştuüvyzwxqABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ0123456789.,;?’@!\-:\'/() ]+')
    total_count = 0
    clean_count = 0

    with open(input_file, 'r', encoding='utf-8') as f, \
            open(output_file, 'w', encoding='utf-8') as f_o, \
            open(removed_file, 'w', encoding='utf-8') as remove_f:

        for line in f:
            line = line.replace('’', '\'')
            line = line.replace('`', '\'')
            line = line.replace('´', '\'')
            line = line.replace('‘', '\'')
            line = re.sub(r'\s+', ' ', line).strip()

            if len(line) == 0:
                remove_f.write(f'Removed: {line}\n')
                remove_f.write('Reason: Empty line after cleaning.\n')
                continue
            else:
                words = line.split()
                if len(words) > MAX_SENTENCE_LEN:
                    print(f'This line has reached max len: {len(line)}')
                    remove_f.write(f'Removed: {line}\n')
                    remove_f.write(f'Reason: Line has reached max len: {len(line)}\n')
                    continue

                ignore_line = False
                for word in words:
                    if len(word) > 45:
                        print(f'This line contains word that exceeds max length: {word} -> {len(word)}')
                        ignore_line = True
                        break

                if ignore_line:
                    remove_f.write(f'Removed: {line}\n')
                    remove_f.write(f'Reason: Line contains word bigger than 45: {word}\n')
                else:
                    if not pattern.fullmatch(line):
                        remove_f.write(f'Removed: {line}\n')
                        problematic_chars = ", ".join(set(line) - set(pattern.pattern))
                        remove_f.write(f'Reason: Problematic characters: {problematic_chars}\n')
                    else:
                        f_o.write(line + '\n')
                        clean_count += 1

                total_count += 1

    print(f'Removed lines: {total_count - clean_count}')
    print(f'Included lines: {clean_count}')
    print(f'Total lines: {total_count}')


# here only we removed unwanted character lines.
file_path = '../source/original-data/testkumesi-gs.txt'
clean_file_path = '../source/original-data/testkumesi-gs-norm.txt'
removed_f = '../source/original-data/test-removed.txt'
clean_file(file_path, clean_file_path, removed_f)
