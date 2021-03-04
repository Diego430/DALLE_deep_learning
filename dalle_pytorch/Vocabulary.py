# adapted from https://www.kdnuggets.com/2019/11/create-vocabulary-nlp-tasks-python.html

class Vocabulary:
    #PAD_token = 0   # Used for padding short sentences
    #SOS_token = 1   # Start-of-sentence token
    #EOS_token = 2   # End-of-sentence token

    def __init__(self, name):
        self.name = name  # something by which to refer to our Vocabulary object
        self.word2index = {}  # dictionary to hold word token to corresponding word index values
        self.word2count = {}  # dictionary to hold individual tokens (word) counts
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}  # dictionary holding the reverse of word2index, special tokens added right away
        self.num_words = 3  # count of the number of tokens (words)
        self.num_sentences = 0  # count of the number of text chunks of any indiscriminate length (sentences)
        self.longest_sentence = 0  # length of the longest corpus sentence by number of tokens

    def add_word(self, word):
        word = word.lower().replace('.', ' EOS ').replace(',','')
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in sentence.split(' '):
            sentence_len += 1
            self.add_word(word)
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        word = word.lower().replace('.', '')
        return self.word2index[word]

    def sentence_to_index(self, sentence):
        sentence_len = 0
        sentence_codes = []
        for word in sentence.split(' '):
            sentence_len += 1
            sentence_codes.append(self.to_index(word))
        return sentence_codes