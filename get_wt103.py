import os
from datasets import load_dataset
from tqdm import tqdm

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main():
    print("=== Downloading WikiText-103 ===")
    
    # Create data directory
    data_dir = 'data/wikitext-103'
    ensure_dir(data_dir)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # Save splits
    def save_split(split_name):
        print(f"Processing {split_name} split...")
        with open(os.path.join(data_dir, f"{split_name}.txt"), 'w', encoding='utf-8') as f:
            for item in tqdm(dataset[split_name]):
                if item['text'].strip():  # Only write non-empty lines
                    f.write(item['text'].strip() + '\n')
    
    save_split('train')
    save_split('validation')
    save_split('test')
    
    print("\nDone! Dataset saved to", data_dir)

if __name__ == '__main__':
    main() 