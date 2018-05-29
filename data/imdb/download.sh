#!/bin/bash

# Download IMDB dataset
echo "Downloading..."
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O imdb.tar.gz
tar -xf imdb.tar.gz


# Prepare dataset
echo "Prepare for IMDB"
find aclImdb/train/pos/ -name "*.txt" | sort > imdb_train_pos_list.txt
find aclImdb/train/neg/ -name "*.txt" | sort > imdb_train_neg_list.txt
find aclImdb/test/pos/ -name "*.txt" | sort > imdb_test_pos_list.txt
find aclImdb/test/neg/ -name "*.txt" | sort > imdb_test_neg_list.txt
find aclImdb/train/unsup/ -name "*.txt" | sort > imdb_unlabled_list.txt

# run Preprocess script
echo "Prepare script is running..."
python preprocess.py prepare_imdb
