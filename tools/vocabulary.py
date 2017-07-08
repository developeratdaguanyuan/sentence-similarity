import spacy
import csv

def _text_to_wordlist(text, nlp):
    try:
        doc = nlp(unicode(text.strip()))
        return [w.text for w in doc]
    except:
        return

def _read_words():
    nlp = spacy.load('en')
    words = []
    reader = csv.reader(open("../data/train.csv"))
    rows = [row for row in reader]
    for i in range(1, len(rows)):
        question_1 = _text_to_wordlist(rows[i][3].strip(), nlp)
        question_2 = _text_to_wordlist(rows[i][4].strip(), nlp)
        if question_1 is not None:
            words.extend([wd for wd in question_1])
        if question_2 is not None:
            words.extend([wd for wd in question_2])

    reader = csv.reader(open("../data/test.csv"))
    rows = [row for row in reader]
    for i in range(1, len(rows)):
        question_1 = _text_to_wordlist(rows[i][1].strip(), nlp)
        question_2 = _text_to_wordlist(rows[i][2].strip(), nlp)
        if question_1 is not None:
            words.extend([wd for wd in question_1])
        if question_2 is not None:
            words.extend([wd for wd in question_2])

    return words

def _build_vocab(wv_path, vocabulary, wv_short_path):
  wv_file = open(wv_path)
  wv_short_file = open(wv_short_path, 'w')
  exist = set()
  while True:
    line = wv_file.readline().strip()
    if not line:
      break
    tokens = line.strip().split(' ')
    if tokens[0].strip() in vocabulary:
      wv_short_file.write(line + "\n")
      exist.add(tokens[0].strip())
  wv_short_file.close()
  wv_file.close()
  non_exist = vocabulary - exist
  for word in non_exist:
    print(word)

set_1 = set(_read_words())
print(set_1)

_build_vocab('../data/glove.840B.300d.txt', set_1, '../data/glove.840B.300d.short.txt')
