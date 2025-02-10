#!/bin/bash

echo "=== Downloading WikiText-103 ==="
mkdir -p data
cd data

if [[ ! -d 'wikitext-103' ]]; then
    wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip
    unzip -q wikitext-103-v1.zip
    cd wikitext-103
    mv wiki.train.tokens train.txt
    mv wiki.valid.tokens valid.txt
    mv wiki.test.tokens test.txt
    cd ..
fi

echo "Done!" 