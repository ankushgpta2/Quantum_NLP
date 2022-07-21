import os
import sys
from lambeq_functions import *
from lstm_functions import *
from datasets import *
import argparse

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
print(test_sentence)
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}
print(word_to_ix)

# read in parameters
def get_hyperparams():
    args = get_args().parse_args()
    parameters = vars(args)
    return parameters


def get_args():
    # initialize parser
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")
    # general arguments and default values
    parser.add_argument('--flag_for_lambeq_default', type=bool, default=False)
    parser.add_argument('--flag_for_lambeq_news', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_default', type=bool, default=True)
    parser.add_argument('--flag_for_lstm_news', type=bool, default=False)
    return parser


class MainRunner:
    def __init__(self):
        self.parameters = get_hyperparams()
        # lambeq dataset
        D = DataSets()
        D.get_default_lambeq_data()
        self.lambeq_default_data = D.data
        # news data
        D = DataSets()
        D.get_news_data()
        self.news_data = D.data
        # for data_flag
        self.data_flag_default = 'lambeq_default_data'
        self.data_flag_news = 'news_data'

    def run_lambeq_on_default(self):
        # run lambeq on default data
        QP = LambeqProcesses(dataset=self.lambeq_default_data, data_flag=self.data_flag_default)
        QP.main()

    def run_lambeq_on_news(self):
        # run lambeq on news data
        QP = LambeqProcesses(dataset=self.news_data, data_flag=self.data_flag_news)
        QP.main()

    def run_lstm_on_default(self):
        # run LSTM on default data
        LSTMP = RunLSTM(dataset=self.lambeq_default_data, embedding_dim=10, context_size=2)
        LSTMP.main()

    def run_lstm_on_news(self):
        # run LSTM on news data
        LSTMP = RunLSTM(dataset=self.news_data, data_flag=self.data_flag_news)
        LSTMP.main()

    def run_main(self):
        if self.parameters['flag_for_lambeq_default'] is True:
            self.run_lambeq_on_default()

        if self.parameters['flag_for_lambeq_news'] is True:
            self.run_lambeq_on_news()

        if self.parameters['flag_for_lstm_default'] is True:
            self.run_lstm_on_default()

        if self.parameters['flag_for_lstm_news'] is True:
            self.run_lstm_on_news()


M = MainRunner()
M.run_main()
