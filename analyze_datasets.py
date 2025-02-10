import os
from collections import Counter
from tqdm import tqdm
import re

def analyze_file(filepath, name=""):
    print(f"\n=== Analyzing {name} ===")
    
    # Count total words and unique tokens
    word_counter = Counter()
    total_lines = 0
    total_words = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            total_lines += 1
            # Split on whitespace and keep punctuation
            words = re.findall(r'\S+', line)
            word_counter.update(words)
            total_words += len(words)
    
    # Calculate statistics
    vocab_size = len(word_counter)
    most_common = word_counter.most_common(10)
    rare_words = len([w for w, c in word_counter.items() if c == 1])
    
    print(f"\nStatistics:")
    print(f"Total lines: {total_lines:,}")
    print(f"Total words: {total_words:,}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Rare words (freq=1): {rare_words:,} ({rare_words/vocab_size*100:.1f}% of vocab)")
    print(f"\nMost common tokens:")
    for word, count in most_common:
        print(f"  {word}: {count:,}")
    
    return total_words, vocab_size

def main():
    # Analyze WikiText-103 train
    wt103_words, wt103_vocab = analyze_file('data/wikitext-103/train.txt', "WikiText-103 Train")
    
    # Analyze Estonian train
    et_words, et_vocab = analyze_file('data/estonian/train.txt', "Estonian Train")
    
    # Print comparison
    print("\n=== Comparison ===")
    print(f"WikiText-103: {wt103_words:,} words, {wt103_vocab:,} vocab")
    print(f"Estonian: {et_words:,} words, {et_vocab:,} vocab")

if __name__ == '__main__':
    main() 