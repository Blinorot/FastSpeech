import re

import nltk


def phonemizer():
    nltk.download('cmudict')
    corpus = nltk.corpus.cmudict.dict()

    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space",
    ]

    corpus['kantorovich'] = ['K AA0 N T AO R AH V IH1 CH'.split()] # add word to lexicon
    
    phoneme_tests = []

    for line in tests:
        line = line.lower()
        phoneme_line = []
        for word in line.split():
            word = re.sub(r'[^\w\s]', '', word)
            phoneme = corpus[word][0]
            phoneme_line.extend(phoneme)
        phoneme_line = ' '.join(phoneme_line)
        phoneme_tests.append(phoneme_line)

    for line in phoneme_tests:
        print(line)


if __name__ == '__main__':
    phonemizer()
