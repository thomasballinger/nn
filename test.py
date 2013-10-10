import random
from fmtstr.fmtfuncs import red, green
from functools import wraps
from itertools import repeat
from pprint import pprint

data = {'spanish':['hola', 'iglesias', 'tonto', 'biblioteca'],
        'english':['hello', 'church', 'stupid', 'library'],
        'nonsense':['qwerty', 'uiop', 'asdf', 'zxcv']}

def normalized(func):
    """Makes the domain of a function into 0 to 1"""
    max_value = max(func(word) for dataset in data.values() for word in dataset)
    min_value = min(func(word) for dataset in data.values() for word in dataset)
    @wraps(func)
    def new_func(word):
        return (func(word) - min_value) / (max_value - min_value)
    return new_func

@normalized
def ratio_vowels(word):
    return len([x for x in word if x in 'aeiou']) / float(len(word))

@normalized
def num_letters(word):
    return len(word)

def ratio_of_letters(letter):
    @normalized
    def ratio_letters(word):
        return word.lower().count(letter) / float(len(word))
    ratio_letters.__name__ = "ratio_of_'%s's" % letter
    return ratio_letters

def random_weights(*funcs):
    return {func:2*random.random() - 1 for func in funcs}

def neuron(weights, threshold=0):
    """Returns a function which evaluates a word based"""
    def func(word):
        return sum(func(word)*weight for func, weight in weights.items()) > threshold
    return func

def evaluate(test_neuron, dataset):
    return [test_neuron(word) == answer for word, answer in dataset].count(True) / float(len(dataset))

def show_correct(test_neuron, dataset):
    print '\n'.join(["%s: %s" % (word, green('correct') if test_neuron(word) == answer else red('incorrect')) for word, answer in dataset])

def find_best_weights(n, dataset, funcs):
    return max((evaluate(neuron(weights), dataset), weights) for weights in [random_weights(*funcs) for _ in range(n)])


test_data = zip(data['english'], repeat(True)) + zip(data['spanish'], repeat(False)) + zip(data['nonsense'], repeat(False))

def one_example():
    english_detector = neuron({ratio_vowels: 1, num_letters: -.1, ratio_of_letters('a'): 1})
    print evaluate(english_detector, test_data)
    show_correct(english_detector, test_data)

score, weights = find_best_weights(1000, test_data,
        [ratio_vowels,
         num_letters,
         ratio_of_letters('a'),
         ratio_of_letters('q'),
         ratio_of_letters('i'),
         ratio_of_letters('o'),
         ratio_of_letters('u'),
         ratio_of_letters('w'),
        ])

print score
pprint(weights)
show_correct(neuron(weights), test_data)

