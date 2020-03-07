import math, random

################################################################################
# Part 0: Utility Functions
################################################################################

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''
    text = start_pad(n) + text
    grams = []
    for index in range(n, len(text)):
        char = text[index]
        context = text[(index - n) : index]
        gram = (context, char)
        grams.append(gram)
    return grams

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
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
        self.grams = {}
        self.contexts = {}

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        grams = ngrams(self.n, text)
        for gram in grams:
            context, char = gram
            if context in self.grams:
                if char in self.grams[context]:
                    self.grams[context][char] += 1
                else:
                    self.grams[context][char] = 1
            else:
                self.grams[context] = {}
                self.grams[context][char] = 1

            if context in self.contexts:
                self.contexts[context] += 1
            else:
                self.contexts[context] = 1

            if char not in self.vocab:
                self.vocab.add(char)
        self.sorted_vocab = sorted(list(self.vocab))

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        if context in self.grams:
            return (self.grams[context].get(char, 0) + self.k) / (self.contexts.get(context, 1e-10) + self.k*len(self.vocab))
        else:
            return (1 + self.k) / (len(self.vocab) + self.k*len(self.vocab))

    def random_char(self, context):
        ''' Returns a random character based on the given context and the
            n-grams learned by this model '''
        #random.seed(1)
        if context not in self.contexts:
            r = random.random()
            index = int(r * len(self.vocab))
            return self.sorted_vocab[index]
        if len(context) > self.n:
            context = context[(len(context) - n): ]

        r = random.random()
        running_prob = 0.0
        for char in self.sorted_vocab:
            char_prob = self.prob(context, char)
            if r >= running_prob and r < (running_prob + char_prob):
                return char
            running_prob += char_prob

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        context = start_pad(self.n)
        output_text = ""
        for i in range(length):
            next_char = self.random_char(context)
            context = context[1:] + next_char
            output_text += next_char
        return output_text

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        N = len(text)
        score = 0.0
        text = start_pad(self.n) + text

        for index in range(self.n, len(text)):
            context = text[(index - self.n) : index]
            char = text[index]
            ngram_prob = self.prob(context, char)
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
        ''' Returns the set of characters in the vocab '''
        return self.vocab

    def update(self, text):
        ''' Updates the model n-grams based on text '''
        for n in range(self.n + 1):
            grams = ngrams(n, text)
            for gram in grams:
                context, char = gram
                if context in self.grams:
                    if char in self.grams[context]:
                        self.grams[context][char] += 1
                    else:
                        self.grams[context][char] = 1
                else:
                    self.grams[context] = {}
                    self.grams[context][char] = 1

                if context in self.contexts:
                    self.contexts[context] += 1
                else:
                    self.contexts[context] = 1

                if char not in self.vocab:
                    self.vocab.add(char)

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        prob = 0.0
        for n in range(self.n + 1):
            shortened_context = context[(len(context) - n) : ] # context length = 0, 1, ..., self.n
            if shortened_context in self.grams:
                prob += self.lambdas[n] * (self.grams[shortened_context].get(char, 0) + self.k) / (self.contexts[shortened_context] + self.k*len(self.vocab))
            else:
                prob += self.lambdas[n] * ((1 + self.k) / (len(self.vocab) + self.k*len(self.vocab)))
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

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([1, 0, 0, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 1, 0, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 1, 0, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 1, 0, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 0, 1, 0])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0, 0, 0, 0, 0, 1])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.01, 0.01, 0.01, 0.01, 0.9, 0.06])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.05, 0.75, 0.05])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.1, 0.1, 0.1, 0.1, 0.4, 0.2])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.01)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.1)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.2)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=0.5)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=1)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))

    m = create_ngram_model(NgramModelWithInterpolation, 'shakespeare_input.txt', 5, k=2)
    m.set_lambdas([0.05, 0.05, 0.05, 0.1, 0.5, 0.25])
    with open('shakespeare_sonnets.txt', encoding='utf-8', errors='ignore') as f:
        print(m.perplexity(f.read()))
