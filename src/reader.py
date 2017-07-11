import spacy
import csv
import os
import numpy as np


# magic feature for Quora Question pairs
def _read_magic_feature(filename):
    reader = open(filename)
    data = list()
    lines = reader.readlines()
    for line in lines:
        tokens = line.strip().split(' ')
        data.append([float(t) for t in tokens])
    reader.close()

    return data


def _load_vocabulary(path):
    word_to_id = dict()
    wv_file = open(path)
    while True:
        line = wv_file.readline()
        if not line:
            break
        tokens = line.strip().split()
        word_to_id[tokens[0]] = len(word_to_id) + 1 # Start from 1
    wv_file.close()

    return word_to_id


def _text_to_wordlist(text, nlp):
    try:
        doc = nlp(unicode(text.strip()))
        return [w.text for w in doc]
    except:
        return


def _build_data(word_to_id, filename, magic_feature_filename):
    nlp = spacy.load('en')

    data = []
    reader = csv.reader(open(filename))
    rows = [row for row in reader]
    magic_feature_rows = _read_magic_feature(magic_feature_filename)

    for i in range(1, len(rows)):
        question_1 = _text_to_wordlist(rows[i][3].strip(), nlp)
        question_2 = _text_to_wordlist(rows[i][4].strip(), nlp)

        if question_1 is not None and question_2 is not None:
            wd_list = [wd for wd in question_1]
            question1 = [word_to_id[wd] if wd in word_to_id else 0 for wd in wd_list]
            wd_list = [wd for wd in question_2]
            question2 = [word_to_id[wd] if wd in word_to_id else 0 for wd in wd_list]
            if len(question1) == 0 or len(question2) == 0:
                continue

            label = int(rows[i][5])
            data.append({'question1': question1,
                         'question2': question2,
                         'magic_feature': magic_feature_rows[i - 1],
                         'label': label})

    return data


def _initialize_random_matrix(shape, scale=0.05, seed=0):
    if len(shape) != 2:
        raise ValueError("Shape of embedding matrix must be 2D, "
                         "got shape {}".format(shape))
    numpy_rng = np.random.RandomState(seed)

    return numpy_rng.uniform(low=-scale, high=scale, size=shape)


def _get_word_embedding(wv_path, word_to_id):
    m = _initialize_random_matrix((1 + len(word_to_id), 300))
    cnt = 0
    wv_file = open(wv_path)
    while True:
        line = wv_file.readline()
        if not line:
            break
        cnt += 1
        tokens = line.strip().split(' ')
        if word_to_id[tokens[0].strip()] != cnt:
            print('ERROR')
            break
        m[cnt] = [np.float32(tokens[i]) for i in range(1, len(tokens))]
    wv_file.close()

    return m


def build_data(data_dir):
    # preparing path
    wv_path = os.path.join(data_dir, 'glove.840B.300d.short.txt')
    file_path = os.path.join(data_dir, 'train.csv')
    magic_feature_path = os.path.join(data_dir, 'feature_magic_train.txt')

    # build data
    word_to_id = _load_vocabulary(wv_path)
    data = _build_data(word_to_id, file_path, magic_feature_path)
    train_data = data[:int(len(data) * 0.8)]
    valid_data = data[int(len(data) * 0.8) : int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]

    # load word embeddings
    word_embedding = _get_word_embedding(wv_path, word_to_id)

    return {'train_data': train_data, 'valid_data': valid_data,
            'test_data': test_data, 'word_embedding': word_embedding}


class DataProducer(object):
    def __init__(self, data, cycle=True):
        self.question1 = [d['question1'] for d in data]
        self.question2 = [d['question2'] for d in data]
        self.label = [d['label'] for d in data]

        self.cycle = cycle
        self.size = len(data)
        self.cursor = 0

    def next(self, n):
        if (self.cursor + n - 1 >= self.size):
            self.cursor = 0
            if self.cycle is False:
                return None
        curr_question1 = self.question1[self.cursor:self.cursor+n]
        curr_question2 = self.question2[self.cursor:self.cursor+n]
        curr_label = self.label[self.cursor:self.cursor+n]
        self.cursor += n

        length_1 = [len(l) for l in curr_question1]
        length_2 = [len(l) for l in curr_question2]
        max_length_1 = max(l for l in length_1)
        max_length_2 = max(l for l in length_2)

        x_1 = np.zeros([n, max_length_1], dtype=np.int32)
        x_2 = np.zeros([n, max_length_2], dtype=np.int32)
        len_1 = np.array(length_1)
        len_2 = np.array(length_2)
        l = np.zeros([n, 1])

        for i, x_i in enumerate(x_1):
            x_i[:len(curr_question1[i])] = np.array(curr_question1[i])
        for i, x_i in enumerate(x_2):
            x_i[:len(curr_question2[i])] = np.array(curr_question2[i])

        for i, l_i in enumerate(l):
            l_i[0] = curr_label[i]

        return x_1, x_2, len_1, len_2, l


if __name__ == "__main__":
    ret = build_data('../data')
    train_data = ret['train_data']
    valid_data = ret['valid_data']
    test_data = ret['test_data']
    word_embedding = ret['word_embedding']

    print(word_embedding[0])
    print(word_embedding[1])

    print(len(test_data))
    test_data_provider = DataProducer(test_data, cycle=False)

    cnt = 0
    while True:
        data = test_data_provider.next(1000)
        if data is not None:
            cnt += 1
            print(cnt * 1000)
        else:
            break;
