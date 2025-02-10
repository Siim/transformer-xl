import os
import sys
import json
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from collections import Counter
import torch
from utils.vocabulary import Vocab
import re

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def clean_text(text):
    """Clean and normalize text similar to WikiText-103 format, but without Wikipedia-specific formatting."""
    # Replace hyphens with @-@ like in WikiText-103
    text = re.sub(r'(?<=[^\s])-(?=[^\s])', ' @-@ ', text)
    
    # Add spaces around punctuation but handle special cases
    text = re.sub(r'([.,!?()"])', r' \1 ', text)
    
    # Handle quotes consistently
    text = re.sub(r'``', ' " ', text)
    text = re.sub(r"''", ' " ', text)
    
    # Handle numbers with commas
    text = re.sub(r'(\d+),(\d+)', r'\1@,@\2', text)
    
    # Handle special number cases after punctuation spacing
    text = re.sub(r'(\d+) \. (\d+)', r'\1.\2', text)  # Rejoin decimal numbers
    text = re.sub(r'(\d+) @,@ (\d+)', r'\1,\2', text)  # Rejoin numbers with commas
    
    # Handle URLs (if they appear in your corpus)
    text = re.sub(r'https?://\S+', ' <URL> ', text)
    
    # Handle emails (if they appear in your corpus)
    text = re.sub(r'\S+@\S+\.\S+', ' <EMAIL> ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def estimate_words_per_line(dataset, sample_size=1000):
    """Estimate average words per line to determine how many lines we need."""
    sample = dataset.select(range(min(len(dataset), sample_size)))
    total_words = 0
    for item in sample:
        words = re.findall(r'\S+', item['text'])
        total_words += len(words)
    return total_words / len(sample)

def main():
    # Create data directory
    data_dir = '../data/estonian/'
    ensure_dir(data_dir)
    
    print('Loading Estonian corpus...')
    dataset = load_dataset("siimh/estonian_corpus_2021", 
                         data_files="corpus_et_clean.jsonl",
                         split="train")
    
    # Estimate how many lines we need to match WikiText-103's ~101M words
    target_words = 101_425_671  # From WikiText-103 analysis
    words_per_line = estimate_words_per_line(dataset)
    target_lines = int(target_words / words_per_line)
    
    print(f"\nEstimated words per line: {words_per_line:.1f}")
    print(f"Target total lines to match WikiText-103: {target_lines:,}")
    
    # Sample the required number of lines
    if len(dataset) > target_lines:
        indices = np.random.choice(len(dataset), target_lines, replace=False)
        dataset = dataset.select(indices)
        print(f"Sampled dataset size: {len(dataset):,} entries")
    
    # Split into train/valid/test (same proportions as WikiText-103)
    # WikiText-103 has ~1.17M total lines: 1.165M train, 2.9K test, 2.5K valid
    valid_size = 2500
    test_size = 3000
    train_size = len(dataset) - valid_size - test_size
    
    # Create splits
    splits = dataset.train_test_split(test_size=valid_size + test_size)
    train_dataset = splits['train']
    temp_splits = splits['test'].train_test_split(test_size=test_size)
    valid_dataset = temp_splits['train']
    test_dataset = temp_splits['test']
    
    print(f"\nSplit sizes:")
    print(f"Train: {len(train_dataset):,}")
    print(f"Valid: {len(valid_dataset):,}")
    print(f"Test: {len(test_dataset):,}")
    
    # First collect all tokens to build vocabulary
    print('\nCollecting all tokens for vocabulary...')
    all_tokens = Counter()
    for item in tqdm(dataset, desc='Processing all text'):
        text = clean_text(item['text'])
        if text:
            tokens = text.split()
            all_tokens.update(tokens)
            all_tokens.update(['<eos>'])  # Add EOS token for each line
    
    # Initialize vocabulary with all tokens (no min_freq or max_size like wt103)
    vocab = Vocab(special=['<eos>'], 
                 lower_case=False,  # Estonian is case-sensitive
                 delimiter=' ')     # Split on spaces after cleaning
    
    # Add all tokens to vocabulary counter
    vocab.counter.update(all_tokens)
    
    # Build vocabulary before saving splits
    print('\nBuilding vocabulary...')
    vocab.build_vocab()
    print('Vocabulary size:', len(vocab))
    
    # Process and save each split
    def save_split(dataset, filename):
        print(f'\nProcessing {filename}...')
        with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as f:
            for item in tqdm(dataset, desc=f'Writing {filename}'):
                # Clean and normalize text
                text = clean_text(item['text'])
                if text:  # Only write non-empty lines
                    f.write(text + '\n')
    
    # Save splits
    print('\nSaving splits...')
    save_split(train_dataset, 'train.txt')
    save_split(valid_dataset, 'valid.txt')
    save_split(test_dataset, 'test.txt')
    
    # Save vocabulary size info
    with open(os.path.join(data_dir, 'vocab_size.txt'), 'w') as f:
        f.write(str(len(vocab)))
    
    print('\nDone! Dataset saved to', data_dir)
    print('\nYou can now use this dataset with transformer-xl by specifying:')
    print('  --data ../data/estonian/')
    print('  --dataset wt103')  # We'll use wt103 settings since it's similar in nature

if __name__ == '__main__':
    main() 