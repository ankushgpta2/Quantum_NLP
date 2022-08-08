import os
import argparse
import numpy as np
from datasets import DataSets
from lambeq_functions import LambeqProcesses
from lstm_functions import RunLSTM

"""
try:
    import lightrun
    lightrun.enable(company_key='ea342b0a-669a-4e63-94f6-1a412dfb2b98')
except ImportError as e:
    print("Error importing Lightrun: ", e)
"""


# read in parameters
def get_hyperparams():
    """ Gets the hyperparameters from the argparser
    """
    args = get_args().parse_args()
    parameters = vars(args)
    return parameters


def get_args():
    """ Expected args from user and default values associated with them
    """
    # initialize parser
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")
    # which models and datasets to run
    parser.add_argument('--flag_for_lambeq_default', type=bool, default=False) # good
    parser.add_argument('--flag_for_lambeq_news', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_default', type=bool, default=False) # good
    parser.add_argument('--flag_for_lstm_news', type=bool, default=False) # good
    parser.add_argument('--flag_for_lambeq_corona', type=bool, default=False) # good
    parser.add_argument('--flag_for_lstm_corona', type=bool, default=False) # good
    parser.add_argument('--flag_for_lambeq_ecommerce', type=bool, default=True)
    parser.add_argument('--flag_for_lstm_ecommerce', type=bool, default=False) # good
    # for the lambeq model
    parser.add_argument('--lambeq_batch_size', type=int, default=16)
    parser.add_argument('--lambeq_epochs', type=int, default=5)
    # for the lstm model
    parser.add_argument('--lstm_batch_size', type=int, default=16)
    parser.add_argument('--lstm_epochs', type=int, default=200)
    parser.add_argument('--lstm_embedding_dim', type=int, default=10)
    parser.add_argument('--lstm_vocab_size', type=int, default=20000)
    parser.add_argument('--lstm_hidden_dim', type=int, default=256)
    parser.add_argument('--lstm_train_split', type=float, default=0.7)
    parser.add_argument('--lstm_val_split', type=float, default=0.2)
    parser.add_argument('--lstm_test_split', type=float, default=0.1)
    parser.add_argument('--lstm_lr', type=float, default=0.0001)
    return parser


class MainRunner:
    def __init__(self):
        self.parameters = get_hyperparams()
        # get vars for training/eval of the models
        self.splits = {'train': self.parameters['lstm_train_split'], 'val': self.parameters['lstm_val_split'],
                       'test': self.parameters['lstm_test_split']}
        try:
            assert np.allclose(sum(self.splits.values()), 1)
        except ValueError:
            print(f'The split sizes for the dataset must equal to one')

        self.news_data_path = '/Users/ankushkgupta/Documents/Quantum_NLP/datasets/news_classification_true_false'
        self.lambeq_default_data_path = '/Users/ankushkgupta/Documents/Quantum_NLP/datasets/lambeq_default_data'
        self.coronavirus_tweets_path = '/Users/ankushkgupta/Documents/Quantum_NLP/datasets/coronavirus_tweets'
        self.ecommerce_data_path = '/Users/ankushkgupta/Documents/Quantum_NLP/datasets/eccomerce_data'

    def run_lambeq_on_default(self):
        """ Runs the lambeq model on the default data set
        """
        D = DataSets(splits=self.splits, model_type='lambeq', path=self.lambeq_default_data_path)
        D.get_default_lambeq_data()
        default_data_for_lambeq = D.data
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=default_data_for_lambeq,
                             data_flag='lambeq_default_data'
                             )
        QP.train()

    def run_lambeq_on_news(self):
        """ Runs the lambeq model on news data set
        """
        D = DataSets(splits=self.splits, model_type='lambeq', path=self.news_data_path)
        D.get_news_data()
        news_data_for_lambeq = D.data
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=news_data_for_lambeq,
                             data_flag='news_data'
                             )
        QP.train()

    def run_lambeq_on_corona(self):
        D = DataSets(splits=self.splits, model_type='lambeq', path=self.coronavirus_tweets_path)
        D.get_coronavirus_tweet_data()
        coronavirus_tweets = D.data
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=coronavirus_tweets,
                             data_flag='coronavirus_tweets'
                             )
        QP.train()

    def run_lambeq_on_ecommerce(self):
        D = DataSets(splits=self.splits, model_type='lambeq', path=self.ecommerce_data_path)
        D.get_ecommerce_data()
        ecommerce_data = D.data
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=ecommerce_data,
                             data_flag='ecommerce'
                             )
        QP.train()

    def run_lstm_on_default(self):
        """ Runs the lstm model on the default data set
        """
        D = DataSets(splits=self.splits, model_type='lstm', path=self.lambeq_default_data_path)
        D.get_default_lambeq_data()
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='lambeq_default_data',
                        full_csv_path=os.path.join(self.lambeq_default_data_path, 'split_data/full.csv'),
                        splits=self.splits
                        )
        LSTMP.run_lstm()

    def run_lstm_on_news(self):
        """ Runs the lstm model on the news data set
        """
        D = DataSets(splits=self.splits, model_type='lstm', path=self.news_data_path)
        D.get_news_data()
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='news_data',
                        full_csv_path=os.path.join(self.news_data_path, 'split_data/full.csv'),
                        splits=self.splits
                        )
        LSTMP.run_lstm()

    def run_lstm_on_corona(self):
        D = DataSets(splits=self.splits, model_type='lstm', path=self.coronavirus_tweets_path)
        D.get_coronavirus_tweet_data()
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='coronavirus_tweets',
                        full_csv_path=os.path.join(self.coronavirus_tweets_path, 'split_data/full.csv'),
                        splits=self.splits
                        )
        LSTMP.run_lstm()

    def run_lstm_on_ecommerce(self):
        D = DataSets(splits=self.splits, model_type='lstm', path=self.ecommerce_data_path)
        D.get_ecommerce_data()
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='ecommerce',
                        full_csv_path=os.path.join(self.ecommerce_data_path, 'split_data/full.csv'),
                        splits=self.splits
                        )
        LSTMP.run_lstm()

    def run_main(self):
        """ Calls the specific function for each model and dataset combination to run
        """
        if self.parameters['flag_for_lambeq_default'] is True:
            self.run_lambeq_on_default()

        if self.parameters['flag_for_lambeq_news'] is True:
            self.run_lambeq_on_news()

        if self.parameters['flag_for_lstm_default'] is True:
            self.run_lstm_on_default()

        if self.parameters['flag_for_lstm_news'] is True:
            self.run_lstm_on_news()

        if self.parameters['flag_for_lstm_corona'] is True:
            self.run_lstm_on_corona()

        if self.parameters['flag_for_lambeq_corona'] is True:
            self.run_lambeq_on_corona()

        if self.parameters['flag_for_lambeq_ecommerce'] is True:
            self.run_lambeq_on_ecommerce()

        if self.parameters['flag_for_lstm_ecommerce'] is True:
            self.run_lstm_on_ecommerce()


M = MainRunner()
M.run_main()

