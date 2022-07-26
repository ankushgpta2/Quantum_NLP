from lambeq_functions import LambeqProcesses
from lstm_functions import RunLSTM
from datasets import DataSets
import argparse

"""
try:
    import lightrun
    lightrun.enable(company_key='ea342b0a-669a-4e63-94f6-1a412dfb2b98')
except ImportError as e:
    print("Error importing Lightrun: ", e)
"""


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
    parser.add_argument('--flag_for_lstm_default', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_news', type=bool, default=True)
    return parser


class MainRunner:
    def __init__(self):
        self.parameters = get_hyperparams()
        # lambeq dataset
        D = DataSets()
        D.get_default_lambeq_data()
        self.default_data_for_lambeq = D.data
        # news data
        D = DataSets()
        D.get_news_data()
        self.news_data_for_lambeq = D.data

    def run_lambeq_on_default(self):
        # run lambeq on default data
        QP = LambeqProcesses(dataset=self.default_data_for_lambeq, data_flag='lambeq_default_data')
        QP.train()

    def run_lambeq_on_news(self):
        # run lambeq on news data
        QP = LambeqProcesses(dataset=self.news_data_for_lambeq, data_flag='news_data')
        QP.train()

    def run_lstm_on_default(self):
        # run LSTM on default data
        LSTMP = RunLSTM(embedding_dim=10, context_size=2, data_flag='lambeq_default_data', vocab_size=20000, batch_size=16, hidden_dim=256, num_classes=2, num_epochs=100)
        LSTMP.train()

    def run_lstm_on_news(self):
        # run LSTM on news data
        LSTMP = RunLSTM(embedding_dim=10, context_size=2, data_flag='news_data', vocab_size=20000, batch_size=16, hidden_dim=256, num_classes=2, num_epochs=100)
        LSTMP.train()

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

