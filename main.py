import os
import argparse
import numpy as np
from datasets import DataSets
from models.lambeq_functions import LambeqProcesses
from models.lstm_functions import RunLSTM
from models.hugging_transformer_functions import HuggingBertModel


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
    # TODO: Running LAMBEQ on few of these datasets leads to issues with BobCat being unable to Parse the text
    parser.add_argument('--datasets_for_lstm', type=list, default=['default'],
                        help='which datasets to run through lstm... must be one of these options:'
                             ' news, default, corona, ecommerce, spam, diabetes'
                        )
    parser.add_argument('--datasets_for_lambeq', type=list, default=['default'],
                        help='which datasets to run through lambeq... must be one of these options:'
                             ' news, default, corona, ecommerce, spam, diabetes'
                        )
    parser.add_argument('--datasets_for_transformer', type=list, default=['default'],
                        help='which datasets to run through transformer... must be one of these options:'
                             ' news, default, corona, ecommerce, spam, diabetes'
                        )
    # for the lambeq model
    parser.add_argument('--lambeq_batch_size', type=int, default=16, help='batch size for lambeq model')
    parser.add_argument('--lambeq_epochs', type=int, default=5, help='# of training epochs for lambeq model')
    parser.add_argument('--lambeq_splits', type=list, default=[0.7, 0.2, 0.1], help='splits for lambeq')
    # for the lstm model
    parser.add_argument('--lstm_batch_size', type=int, default=16, help='batch size for lstm model')
    parser.add_argument('--lstm_epochs', type=int, default=200, help='# of training epochs for lstm model')
    parser.add_argument('--lstm_embedding_dim', type=int, default=512, help='embedding dim for lstm')
    parser.add_argument('--lstm_vocab_size', type=int, default=1024, help='reference corpus')
    parser.add_argument('--lstm_hidden_dim', type=int, default=256, help='# of LSTM units in each layer essentially')
    parser.add_argument('--lstm_splits', type=list, default=[0.7, 0.2, 0.1], help='splits for lstm')
    parser.add_argument('--lstm_lr', type=float, default=0.0001, help='learning rate for lstm')
    # for the hugging face transformer model
    parser.add_argument('--transformer_batch_size', type=int, default=8, help='batch size for transformer')
    parser.add_argument('--transformer_epochs', type=int, default=10, help='training epochs for transformer')
    parser.add_argument('--transformer_lr', type=float, default=2e-5, help='learning rate for transformer')
    parser.add_argument('--transformer_weight_decay', type=float, default=0.001,
                        help='how much to decay weights per epoch I think')
    parser.add_argument('--transformer_metric_for_best_model', type=str, default='accuracy',
                        help='metric to use for determining best model')
    parser.add_argument('--transformer_splits', type=list, default=[0.7, 0.2, 0.1], help='splits for transformer')
    parser.add_argument('--use_pretrained', type=bool, default=False, help='whether to use pretrained model')
    return parser


class MainRunner:
    def __init__(self):
        self.parameters = get_hyperparams()
        # get vars for training/eval of the models
        self.diff_models = ['lstm', 'lambeq', 'transformer']
        self.splits = {'lstm': {}, 'lambeq': {}, 'transformer': {}}
        for model_name in self.diff_models:
            splits = self.parameters[f'{model_name}_splits']
            self.splits[model_name]['train'] = splits[0]
            self.splits[model_name]['val'] = splits[1]
            self.splits[model_name]['test'] = splits[2]

        try:
            for key in self.splits.keys():
                assert np.allclose(sum(self.splits[key].values()), 1)
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
        D = DataSets(splits=self.splits[model_name], model_type=model_name, dataset_name=dataset_key, path=self.data_paths[dataset_key])
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
            QP.run_lambeq()

        elif model_name == 'lstm':
            LSTMP = RunLSTM(parameters=self.parameters,
                            data_flag=dataset_key,
                            root_path=self.ROOT_DIR,
                            splits=self.splits
                            )
            print(f'\n********** RUNNING LSTM ON {dataset_key.upper()} ***********\n')
            LSTMP.run_lstm()

        elif model_name == 'transformer':
            HTP = HuggingBertModel(dataset=dataset,
                                   use_pretrained=False,
                                   parameters=self.parameters,
                                   root_path=self.ROOT_DIR,
                                   data_flag=dataset_key,
                                   )
            print(f'\n********** RUNNING HUGGING TRANSFORMER ON {dataset_key.upper()} ***********\n')
            HTP.run_hugging_bert()

    def run_main(self):
        # check that the dataset names are appropriate
        dataset_args = [self.parameters['datasets_for_lstm'], self.parameters['datasets_for_lambeq'],
                        self.parameters['datasets_for_transformer']]
        for i in range(len(dataset_args)):
            if len(dataset_args[i]) != 0:
                for dataset_name in dataset_args[i]:
                    try:
                        assert dataset_name in list(self.data_paths.keys())
                    except ValueError:
                        raise ValueError(f'{dataset_name} is not a valid dataset name.... must be within this list: '
                                         f'{list(self.data_paths.keys())}')

        # call appropriate model for dataset
        for model in self.diff_models:
            datasets = [x for x in self.parameters.keys() if f'datasets_for_{model}' in x]
            for dataset in self.parameters[datasets[0]]:
                self.run_model(dataset_key=dataset, model_name=model)


M = MainRunner()
M.run_main()

