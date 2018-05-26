
import numpy as np

def convert_to_vocab_id(vocab, pos, neg, convert_vocab=True, ignore_unk=False, ign_eos=False):
    # binary class
    # Positive => 1
    # Negative => 0
    dataset_x = []
    dataset_x_length = []
    dataset_y = []

    def conv(words):
        if ignore_unk:
            return [vocab.get(w, 1) for w in words if w in vocab]
        else:
            return [vocab.get(w, 1) for w in words]

    for words in pos:
        if convert_vocab:
            if ign_eos:
                conv_words = conv(words)
            else:
                conv_words = conv(words) + [0]
            word_ids = np.array(conv_words, dtype=np.int32)  # EOS
        else:
            word_ids = ' '.join(words)
        dataset_x.append(word_ids)
        dataset_x_length.append(len(word_ids))
        dataset_y.append(1)

    for words in neg:
        if convert_vocab:
            if ign_eos:
                conv_words = conv(words)
            else:
                conv_words = conv(words) + [0]
            word_ids = np.array(conv_words, dtype=np.int32)  # EOS
        else:
            word_ids = ' '.join(words)
        dataset_x.append(word_ids)
        dataset_x_length.append(len(word_ids))
        dataset_y.append(0)

    dataset_y = np.array(dataset_y, dtype=np.int32)
    return dataset_x, dataset_x_length, dataset_y

def load_file_preprocess(filename, lower=True):
    dataset = []
    def conv(w):
        if lower:
            return w.lower()
        return w
    with open(filename, 'r') as f:
        for l in f:
            words = [conv(w) for w in l.strip().split(' ')]
            dataset.append(words)
    return dataset

def load_dataset_imdb(include_pretrain=False, convert_vocab=True, lower=True,
                      min_count=0, ignore_unk=False, use_semi_data=False,
                      add_labeld_to_unlabel=True):
    lm_dataset = None
    imdb_validation_pos_start_id = 10621  # total size: 12499
    imdb_validation_neg_start_id = 10625

    pos_train = load_file_preprocess('data/imdb/imdb_pos_train.txt', lower=lower)
    pos_dev = load_file_preprocess('data/imdb/imdb_pos_dev.txt', lower=lower)

    neg_train = load_file_preprocess('data/imdb/imdb_neg_train.txt', lower=lower)
    neg_dev = load_file_preprocess('data/imdb/imdb_neg_dev.txt', lower=lower)

    if include_pretrain:
        # Pretrain with LM
        unlabled_lm_train = load_file_preprocess('data/imdb/imdb_unlabled.txt', lower=lower)

    pos_test = load_file_preprocess('data/imdb/imdb_pos_test.txt', lower=lower)
    neg_test = load_file_preprocess('data/imdb/imdb_neg_test.txt', lower=lower)

    train_set = pos_train + neg_train
    if include_pretrain:
        # Pretrain with LM
        train_set += unlabled_lm_train

    word_nums = [float(len(words)) for words in train_set]
    print('train_set:{}'.format(len(train_set)))
    print('avg word number:{}'.format(sum(word_nums) / len(word_nums)))

    vocab = {}
    vocab['<eos>'] = 0  # EOS
    vocab['<unk>'] = 1  # EOS
    word_cnt = {}
    for words in train_set:
        for w in words:
            if lower:
                w = w.lower()
            word_cnt[w] = word_cnt.get(w, 0) + 1
    doc_counts = {}
    for words in train_set:
        doc_seen = set()
        for w in words:
            if w not in doc_seen:
                doc_counts[w] = doc_counts.get(w, 0) + 1
                doc_seen.add(w)

    for words in train_set:
        for w in words:
            if lower:
                w = w.lower()
            if w not in vocab and doc_counts[w] > min_count:
                vocab[w] = len(vocab)
    print('vocab:{}'.format(len(vocab)))

    vocab_limit = {}
    for words in pos_train + neg_train:
        for w in words:
            if lower:
                w = w.lower()
            if w not in vocab_limit and doc_counts[w] > min_count:
                vocab_limit[w] = len(vocab_limit)
    train_vocab_size = len(vocab_limit)

    train_x, train_x_len, train_y = convert_to_vocab_id(vocab, pos_train,
                                                        neg_train, convert_vocab=convert_vocab, ignore_unk=ignore_unk)
    word_nums = [len(x) for x in train_x]
    print('avg word number (train_x): {}'.format(sum(word_nums) / len(word_nums)))
    dev_x, dev_x_len, dev_y = convert_to_vocab_id(
        vocab, pos_dev, neg_dev, convert_vocab=convert_vocab, ignore_unk=ignore_unk)

    word_nums = [len(x) for x in dev_x]
    print('avg word number (dev_x):{}'.format(sum(word_nums) / len(word_nums)))
    test_x, test_x_len, test_y = convert_to_vocab_id(
        vocab, pos_test, neg_test, convert_vocab=convert_vocab, ignore_unk=ignore_unk)

    word_nums = [len(x) for x in test_x]
    print('avg word number (test_x):{}'.format(sum(word_nums) / len(word_nums)))
    dataset = (train_x, train_x_len, train_y,
               dev_x, dev_x_len, dev_y,
               test_x, test_x_len, test_y)
    if include_pretrain:
        lm_train_x, _, _ = convert_to_vocab_id(vocab, unlabled_lm_train, [], ignore_unk=ignore_unk)
        lm_train_all = lm_train_x
        if add_labeld_to_unlabel:
            lm_train_all += train_x

        lm_dev_all = test_x
        lm_train_words_num = sum([len(x) for x in lm_train_all])
        lm_dev_words_num = sum([len(x) for x in lm_dev_all])
        print('lm_words_num:{}'.format(lm_train_words_num))

        lm_train_dataset = np.concatenate(lm_train_all, axis=0).astype(np.int32)
        lm_dev_dataset = np.concatenate(lm_dev_all, axis=0).astype(np.int32)

        lm_dataset = (lm_train_dataset, lm_dev_dataset)
        if use_semi_data:
            lm_train_all_length = [len(x) for x in lm_train_all]
            lm_dataset = (lm_train_all, lm_train_all_length)

    vocab_tuple = (vocab, doc_counts)
    return vocab_tuple, dataset, lm_dataset, train_vocab_size
