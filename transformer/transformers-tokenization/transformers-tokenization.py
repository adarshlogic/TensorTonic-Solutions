class SimpleTokenizer:

    def __init__(self):
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>"
        }

        self.vocab_size = 4

    def build_vocab(self, texts):
        idx = 4

        for text in texts:
            for word in text.split():
                if word not in self.word_to_id:
                    self.word_to_id[word] = idx
                    self.id_to_word[idx] = word
                    idx += 1

        self.vocab_size = len(self.word_to_id)

    def encode(self, text):
        tokens = []

        for word in text.split():
            tokens.append(self.word_to_id.get(word, 1))

        return tokens

    def decode(self, ids):
        words = []

        for i in ids:
            words.append(self.id_to_word.get(i, "<UNK>"))

        return " ".join(words)