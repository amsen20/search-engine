from hazm import *

normalizer = Normalizer()
stemmer = Stemmer()
stopwords = stopwords_list()
stopwords += ['.', '،', ':', ')', '(', 'کشور', '[', ']', 'https', 'ایر', 'farsnews', '«', '»']


class TokenData:
    def __init__(self):
        self.freq = 0
        self.positions = {}

    def add(self, doc_id, pos):
        if doc_id not in self.positions:
            self.positions[doc_id] = set()

        self.positions[doc_id].add(pos)
        self.freq += 1


def get_tokens(content):
    return [
        stemmer.stem(token) for token in list(
            filter(lambda x: x not in stopwords, word_tokenize(
                normalizer.normalize(content)
            ))
        )
    ]
