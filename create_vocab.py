import json
import argparse
from collections import Counter

def build_vocabs(train_file, char_vocab_size=400, word_vocab_size=60000):
    # Read and process the training file
    with open(train_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    char_counter = Counter()
    word_counter = Counter()

    for line in lines:
        text = line.strip()
        char_counter.update(text)
        word_counter.update(text.split())

    # Get the most common characters and words
    most_common_chars = char_counter.most_common(char_vocab_size - 1)
    most_common_words = word_counter.most_common(word_vocab_size - 1)

    # Create vocabularies with <unk> token
    char_vocab = {'<unk>': 0}
    char_vocab.update({char: idx + 1 for idx, (char, _) in enumerate(most_common_chars)})

    word_vocab = {'<unk>': 0}
    word_vocab.update({word: idx + 1 for idx, (word, _) in enumerate(most_common_words)})

    # Save to JSON files
    with open("vocab_char.json", "w", encoding='utf-8') as char_file:
        json.dump(char_vocab, char_file, ensure_ascii=False, indent=4)

    with open("vocab_word.json", "w", encoding='utf-8') as word_file:
        json.dump(word_vocab, word_file, ensure_ascii=False, indent=4)

    print("Vocabularies saved to vocab_char.json and vocab_word.json")

def main():
    parser = argparse.ArgumentParser(description="Build character and word vocabularies from a training file.")
    parser.add_argument('--train-file', type=str, required=True, help='Path to the training file containing text data.')
    args = parser.parse_args()

    build_vocabs(args.train_file)

if __name__ == "__main__":
    main()
