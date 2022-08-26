import os
import argparse
import numpy as np
from datasets import DataSets
from models.lambeq_functions import LambeqProcesses
from models.lstm_functions import RunLSTM


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
    # which models and datasets to run --> MUST BE EXACT NAMES FROM THIS LIST:
    # ['news', 'default', 'corona', 'ecommerce', 'spam', 'diabetes']
    parser.add_argument('--datasets_for_lstm', type=list, default=['default'])
    parser.add_argument('--datasets_for_lambeq', type=list, default=['default'])
    parser.add_argument('--datasets_for_hugging_transformer', type=list, default=[])
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
        self.data_paths = {'news': os.path.join(self.ROOT_DIR, 'datasets/news'),
                           'default': os.path.join(self.ROOT_DIR, 'datasets/default'),
                           'corona': os.path.join(self.ROOT_DIR, 'datasets/corona'),
                           'ecommerce': os.path.join(self.ROOT_DIR, 'datasets/ecommerce'),
                           'spam': os.path.join(self.ROOT_DIR, 'datasets/spam'),
                           'diabetes': os.path.join(self.ROOT_DIR, 'datasets/diabetes'),
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
                                 root_path=self.ROOT_DIR,
                                 )
            print(f'\n********** RUNNING LAMBEQ ON {dataset_key.upper()} ***********\n')
            QP.train()

        elif model_name == 'lstm':
            LSTMP = RunLSTM(parameters=self.parameters,
                            data_flag=dataset_key,
                            root_path=self.ROOT_DIR,
                            splits=self.splits
                            )
            print(f'\n********** RUNNING LSTM ON {dataset_key.upper()} ***********\n')
            LSTMP.run_lstm()

        elif model_name == 'hugging_transformer':
            """
            hold
            """

    def run_main(self):
        # check that the dataset names are appropriate
        dataset_args = [self.parameters['datasets_for_lstm'], self.parameters['datasets_for_lambeq']]
        for i in range(len(dataset_args)):
            if len(dataset_args[i]) != 0:
                for dataset_name in dataset_args[i]:
                    try:
                        assert dataset_name in list(self.data_paths.keys())
                    except ValueError:
                        raise ValueError(f'{dataset_name} is not a valid dataset name.... must be within this list: '
                                         f'{list(self.data_paths.keys())}')

        # call appropriate model for dataset
        for model in ['lstm', 'lambeq', 'hugging_transformer']:
            datasets = [x for x in self.parameters.keys() if f'datasets_for_{model}' in x]
            for dataset in self.parameters[datasets[0]]:
                self.run_model(dataset_key=dataset, model_name=model)


M = MainRunner()
M.run_main()

