import spacy
import csv
import os


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
    return {'train_data': train_data, 'valid_data': valid_data,
            'test_data': test_data, 'word_to_id': word_to_id}

if __name__ == "__main__":
    ret = build_data('../data')
    train_data = ret['train_data']
    valid_data = ret['valid_data']
    test_data = ret['test_data']
    print(len(train_data))
    print(len(valid_data))
    print(len(test_data))
    print(train_data[0]['label'])
    print(train_data[0]['question1'])
    print(train_data[0]['question2'])
    print(train_data[0]['magic_feature'])
