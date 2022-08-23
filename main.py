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
    parser.add_argument('--flag_for_lambeq_default', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lambeq_news', type=bool, default=False)  #TODO: fix issue with lambeq running
    parser.add_argument('--flag_for_lstm_default', type=bool, default=False)  #good
    parser.add_argument('--flag_for_lstm_news', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lambeq_corona', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lstm_corona', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lambeq_ecommerce', type=bool, default=False)  #TODO: fix issue with lambeq running
    parser.add_argument('--flag_for_lstm_ecommerce', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lambeq_spam', type=bool, default=False)  # good
    parser.add_argument('--flag_for_lstm_spam', type=bool, default=True)  # good
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

        self.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_paths = {'news': os.path.join(self.ROOT_DIR, 'datasets/news_classification_true_false'),
                           'default': os.path.join(self.ROOT_DIR, 'datasets/lambeq_default_data'),
                           'corona': os.path.join(self.ROOT_DIR, 'datasets/coronavirus_tweets'),
                           'ecommerce': os.path.join(self.ROOT_DIR, 'datasets/eccomerce_data'),
                           'spam': os.path.join(self.ROOT_DIR, 'datasets/email_spam'),
                           }

    def run_model(self, dataset_key, model_name):
        # get the dataset
        D = DataSets(splits=self.splits, model_type=model_name, dataset_name=dataset_key, path=self.data_paths[dataset_key])
        D.get_dataset()
        dataset = D.data

        # run the appropriate model
        if model_name == 'lambeq':
            QP = LambeqProcesses(parameters=self.parameters,
                                 dataset=dataset,
                                 data_flag=dataset_key,
                                 )
            QP.train()

        elif model_name == 'lstm':
            LSTMP = RunLSTM(parameters=self.parameters,
                            data_flag=dataset_key,
                            full_csv_path=os.path.join(self.data_paths[dataset_key], 'split_data/full.csv'),
                            splits=self.splits
                            )
            LSTMP.run_lstm()

    def run_main(self):
        keys = [x for x in self.parameters.keys() if 'flag' in x]
        vals = [self.parameters[y] for y in keys]
        condition_checks = {}
        
        for x, y in zip(keys, vals): 
            condition_checks[x] = y

        # call appropriate model for dataset
        for flag in condition_checks.keys():
            if condition_checks[flag] is True:
                print(f'\n********** Running {flag.split("_")[-2]} on {flag.split("_")[-1]} ***********\n')
                self.run_model(dataset_key=flag.split("_")[-1], model_name=flag.split("_")[-2])

M = MainRunner()
M.run_main()

