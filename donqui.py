from itertools import repeat
from pprint import pprint

from nn import find_best_weights, neuron, show_correct, ratio_of_letters

def filedata(filename):
    sentences = [s.strip().decode('utf8') for s in open(filename).read().split('\r\n') if s.strip()]
    return sentences

data = (zip(filedata('spanish.txt'), repeat(False)) +
        zip(filedata('english.txt'), repeat(True)))

if __name__ == '__main__':

    score, weights = find_best_weights(10, data,
            [ratio_of_letters('a'),
             ratio_of_letters('i'),
             ratio_of_letters('o'),
             ratio_of_letters('u'),
            ])

    print "%d%% correct" % (score * 100)
    print 'weights (positive number mean more english'
    pprint(weights)
    #show_correct(neuron(weights), data)

