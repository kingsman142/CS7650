import math, random
from typing import List, Tuple

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return ['~'] * n

Pair = Tuple[str, str]
Ngrams = List[Pair]
def ngrams(n, text:str) -> Ngrams:
    text = start_pad(n) + text.strip().split()
    ''' Returns the ngrams of the text as tuples where the first element is
        the n-word sequence (i.e. "I love machine") context and the second is the word '''
    grams = []
    for index in range(n, len(text)):
        word = text[index]
        context = ' '.join(text[(index - n) : index])
        gram = (context, word)
        grams.append(gram)
    return grams

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8') as f:
        model.update(f.read())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab = set()
        self.sorted_vocab = []
        self.count = {}
        self.grams = {}
        self.contexts = {}

    def get_vocab(self):
        ''' Returns the set of words in the vocab '''
        return self.vocab

    def update(self, text:str):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for gram in grams:
            context, word = gram
            if context in self.grams:
                if word in self.grams[context]:
                    self.grams[context][word] += 1
                else:
                    self.grams[context][word] = 1
            else:
                self.grams[context] = {}
                self.grams[context][word] = 1

            if context in self.contexts:
                self.contexts[context] += 1
            else:
                self.contexts[context] = 1

            if word not in self.vocab:
                self.vocab.add(word)
        self.sorted_vocab = sorted(list(self.vocab))

    def prob(self, context:str, word:str):
        ''' Returns the probability of word appearing after context '''
        if context in self.grams:
            return (self.grams[context].get(word, 0) + self.k) / (self.contexts.get(context, 1e-10) + self.k*len(self.vocab))
        else:
            return (1 + self.k) / (len(self.vocab) + self.k*len(self.vocab))

    def random_word(self, context):
        ''' Returns a random word based on the given context and the
            n-grams learned by this model '''
        #random.seed(1)
        if context not in self.contexts:
            r = random.random()
            index = int(r * len(self.vocab))
            return self.sorted_vocab[index]

        r = random.random()
        running_prob = 0.0
        for word in self.sorted_vocab:
            word_prob = self.prob(context, word)
            if r >= running_prob and r < (running_prob + word_prob):
                return word
            running_prob += word_prob

    def random_text(self, length):
        ''' Returns text of the specified word length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        output_text = []
        for i in range(length):
            joined_context = " ".join(context)
            next_word = self.random_word(joined_context)
            context = context[1:]
            context.append(next_word)
            output_text.append(next_word)
        return " ".join(output_text)

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        text = text.split(" ")
        N = len(text)
        score = 0.0
        text = start_pad(self.n) + text

        for index in range(self.n, len(text)):
            context = " ".join(text[(index - self.n) : index])
            word = text[index]
            ngram_prob = self.prob(context, word)
            if ngram_prob == 0:
                return float('inf')
            ngram_prob = math.log(ngram_prob)
            score += ngram_prob

        return math.exp((-(1/N)*score))

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super(NgramModelWithInterpolation, self).__init__(n, k)

        self.lambdas = [1/(n+1)] * (n+1)

    def set_lambdas(self, new_lambdas):
        if len(new_lambdas) == len(self.lambdas) and sum(new_lambdas) == 1.0:
            self.lambdas = new_lambdas

    def get_vocab(self):
        return self.vocab

    def update(self, text:str):
        for n in range(self.n + 1):
            grams = ngrams(n, text)
            for gram in grams:
                context, word = gram
                if context in self.grams:
                    if word in self.grams[context]:
                        self.grams[context][word] += 1
                    else:
                        self.grams[context][word] = 1
                else:
                    self.grams[context] = {}
                    self.grams[context][word] = 1

                if context in self.contexts:
                    self.contexts[context] += 1
                else:
                    self.contexts[context] = 1

                if word not in self.vocab:
                    self.vocab.add(word)

    def prob(self, context:str, word:str):
        context_list = context.split(" ")

        prob = 0.0
        for n in range(self.n + 1):
            shortened_context = " ".join(context_list[(len(context_list) - n) : ]) # context length = 0, 1, ..., self.n
            if shortened_context in self.grams:
                prob += self.lambdas[0] * (self.grams[shortened_context].get(word, 0) + self.k) / (self.contexts[shortened_context] + self.k*len(self.vocab))
            else:
                prob += self.lambdas[0] * ((1 + self.k) / (len(self.vocab) + self.k*len(self.vocab)))
        return prob

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([1/3] * 3)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 3, k=0.01)
    m.set_lambdas([1/4] * 4)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 4, k=0.01)
    m.set_lambdas([1/5] * 5)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([1/6] * 6)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 10, k=0.01)
    m.set_lambdas([1/11] * 11)
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([0, 1, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([0, 0, 1])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([0.9, 0.05, 0.05])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([0.7, 0.15, 0.15])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([0.5, 0.25, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.01)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.1)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.2)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=0.5)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=1.0)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 2, k=2.0)
    m.set_lambdas([1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
