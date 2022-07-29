import argparse
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
    parser.add_argument('--flag_for_lambeq_default', type=bool, default=False)
    parser.add_argument('--flag_for_lambeq_news', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_default', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_news', type=bool, default=False)
    # for the lambeq model
    parser.add_argument('--lambeq_batch_size', type=int, default=16)
    parser.add_argument('--lambeq_epochs', type=int, default=40)
    # for the lstm model
    parser.add_argument('--lstm_batch_size', type=int, default=16)
    parser.add_argument('--lstm_epochs', type=int, default=100)
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
        # get lambeq dataset
        D = DataSets()
        D.get_default_lambeq_data()
        self.default_data_for_lambeq = D.data
        # get news data
        D = DataSets()
        D.get_news_data()
        self.news_data_for_lambeq = D.data
        # get vars for training/eval of the models
        self.splits = {'train': self.parameters['lstm_train_split'], 'val': self.parameters['lstm_val_split'],
                       'test': self.parameters['lstm_test_split']}

    def run_lambeq_on_default(self):
        """ Runs the lambeq model on the default data set
        """
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=self.default_data_for_lambeq,
                             data_flag='lambeq_default_data'
                             )
        QP.train()

    def run_lambeq_on_news(self):
        """ Runs the lambeq model on news data set
        """
        QP = LambeqProcesses(parameters=self.parameters,
                             dataset=self.news_data_for_lambeq,
                             data_flag='news_data'
                             )
        QP.train()

    def run_lstm_on_default(self):
        """ Runs the lstm model on the default data set
        """
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='lambeq_default_data',
                        splits=self.splits
                        )
        LSTMP.run_lstm()

    def run_lstm_on_news(self):
        """ Runs the lstm model on the news data set
        """
        LSTMP = RunLSTM(parameters=self.parameters,
                        data_flag='news_data',
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


M = MainRunner()
M.run_main()

