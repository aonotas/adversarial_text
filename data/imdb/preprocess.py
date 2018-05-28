
import re

def split_by_punct(segment):
    """Splits str segment by punctuation, filters our empties and spaces."""
    return [s for s in re.split(r'\W+', segment) if s and not s.isspace()]

def read_text(filename):
    with open(filename) as f:
        sentences = []
        for line in f:
            line = line.strip()
            words = ' '.join(split_by_punct(line)).strip()
            sentences.append(words)
        return ' '.join(sentences).strip()


def load_file(data_file, split_idx):
    train = []
    dev = []
    with open(data_file) as f:
        for filename in f:
            idx = int(filename.split('/')[-1].split('_')[0])
            words = read_text(filename.strip())
            if idx >= split_idx:
                dev.append(words)
            else:
                train.append(words)
    return train, dev


def load_test_dataset(data_file):
    test = []
    with open(data_file) as f:
        for filename in f:
            words = read_text(filename.strip())
            test.append(words)
    return test

def prepare_imdb():
    # this split is used at
    # https://github.com/tensorflow/models/tree/master/research/adversarial_text
    imdb_validation_pos_start_id = 10621  # total size: 12499
    imdb_validation_neg_start_id = 10625

    def fwrite_data(filename, sentences):
        with open(filename, 'w') as f:
            for words in sentences:
                # line = ' '.join(words)
                line = words
                f.write(line.strip() + '\n')
            f.close()

    pos_train, pos_dev = load_file('imdb_train_pos_list.txt',
                                   imdb_validation_pos_start_id)
    neg_train, neg_dev = load_file('imdb_train_neg_list.txt',
                                   imdb_validation_neg_start_id)

    pos_test = load_test_dataset('imdb_test_pos_list.txt')
    neg_test = load_test_dataset('imdb_test_neg_list.txt')

    fwrite_data('imdb_pos_train.txt', pos_train)
    fwrite_data('imdb_pos_dev.txt', pos_dev)
    fwrite_data('imdb_neg_train.txt', neg_train)
    fwrite_data('imdb_neg_dev.txt', neg_dev)

    fwrite_data('imdb_pos_test.txt', pos_test)
    fwrite_data('imdb_neg_test.txt', neg_test)

    unlabled_lm_train, _ = load_file('imdb_unlabled_list.txt', 100000000)

    fwrite_data('imdb_unlabled.txt', unlabled_lm_train)
    print('Done')

if __name__ == '__main__':
    import sys
    action = sys.argv[1]
    if action == 'prepare_imdb':
        prepare_imdb()
