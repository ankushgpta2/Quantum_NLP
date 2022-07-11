import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# lambeq related packages
from lambeq import BobcatParser, remove_cups, AtomicType, IQPAnsatz, TketModel, QuantumTrainer, SPSAOptimizer, \
    Dataset
from pytket.extensions.qiskit import AerBackend
# torch packages
import torch
from torchtext.data import Field, TabularDataset, BucketIterator
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
# other core packages 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class LambeqProcesses(object):
    def __init__(self, data_flag):
        self.reader = BobcatParser(verbose='text')
        self.ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=1, n_single_qubit_params=3)
        self.backend = AerBackend()
        self.model = TketModel
        self.loss_function = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
        self.acc_function = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
        self.epochs = 10 
        self.optimizer = SPSAOptimizer
        self.batch_size = 30
        self.datasets = DataSets()
        self.data_flag = data_flag

    def main(self):
        # read in all of the data
        if self.data_flag == 'lambeq_default':
            self.datasets.get_default_lambeq_data()
        elif self.data_flag == 'news_data':
            self.datasets.get_news_data()
        # use parser and circuit creator
        self.get_diagram_and_circuit()
        # preparation stuff for training 
        train_dataset, val_dataset, trainer, model = self.prep_for_train()
        # run the training on generated circuit representations + labels
        trainer.fit(train_dataset, val_dataset, logging_step=12)
        # plot the performance
        self.plot_performance(trainer, model)

    # for the parser 
    def get_diagram_and_circuit(self):
        for key in self.datasets.data.keys():
            raw_diagram = self.reader.sentences2diagrams(self.datasets.data[key]['text'])
            diagram = [remove_cups(diagram) for diagram in raw_diagram]
            circuit = [self.ansatz(sub_diagram) for sub_diagram in diagram]
            # diagram[0].draw()
            # circuit[0].draw()
            self.datasets.data[key]['diagram'] = diagram
            self.datasets.data[key]['circuit'] = circuit

    # intializing necessary things for training 
    def prep_for_train(self):
        backend_config = {
            'backend': self.backend,
            'compilation': self.backend.default_compilation_pass(2),
            'shots': 8192
        }
        model = self.model.from_diagrams(self.datasets.data['train']['circuit']+self.datasets.data['dev']
                        ['circuit']+self.datasets.data['test']['circuit'], backend_config=backend_config) 
        trainer = QuantumTrainer(
            model,
            loss_function=self.loss_function,
            epochs=self.epochs,
            optimizer=self.optimizer,
            optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*self.epochs},
            evaluate_functions={'acc': self.acc_function},
            evaluate_on_train=True,
            verbose = 'text',
            seed=0
        )
        train_dataset = Dataset(self.datasets.data['train']['circuit'], self.datasets.data['train']['labels'], 
                        batch_size=self.batch_size)
        val_dataset = Dataset(self.datasets.data['dev']['circuit'], self.datasets.data['dev']['labels'], 
                        shuffle=False)
        return train_dataset, val_dataset, trainer, model

    # getting the plots for model performance on dataset
    def plot_performance(self, trainer, model):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
        ax1.plot(trainer.train_epoch_costs, color='blue')
        ax2.plot(trainer.val_costs, color='red')
        ax3.plot(trainer.train_results['acc'], color='blue')
        ax4.plot(trainer.val_results['acc'], color='red')
        # labels
        ax1.set_title('Training Set'), ax2.set_title('Development Set'), ax3.set_xlabel('Epochs'),
        ax4.set_xlabel('Epochs'), ax1.set_ylabel('Loss'), ax3.set_ylabel('Accuracy')
        plt.suptitle('Loss and Accuracy Performance on Train/Dev Sets', fontweight='bold', fontsize=14)
        if os.path.isdir('figures') is False:
            os.mkdir('figures')
        plt.savefig('figures/loss_acc_lambeq.png')
        test_acc = self.acc_function(model(self.datasets.data['test']['circuit']), 
                        self.datasets.data['test']['labels'])
        print('Test accuracy:', test_acc)
        plt.show()


class DataSets():
    def __init__(self):
        self.data = {'train': {}, 'dev': {}, 'test': {}}

    def get_default_lambeq_data(self):
        """ Text Classification
        """
        # explictly specify the data
        data_paths = ['datasets/mc_train_data.txt', 'datasets/mc_dev_data.txt', 'datasets/mc_test_data.txt']
        for path in data_paths:
            labels, sentences = [], []
            with open(path) as f:
                for line in f:
                    t = float(line[0])
                    labels.append([t, 1-t])
                    sentences.append(line[1:].strip())
            self.place_data_in_dict(path=path, labels=labels, text=sentences)

    def get_news_data(self):
        """ Text Classification
        """
        # read the data from .csv file
        data_paths = ['datasets/news_classification_true_false/train.csv', 'datasets/news_classification_true_false/valid.csv', 
                            'datasets/news_classification_true_false/test.csv']
        for path in data_paths:
            df = pd.read_csv(path)
            self.place_data_in_dict(path=path, text=df['text'][:].to_list(), labels=df['label'][:].to_list())

    def get_squad_data(self):
        """ Text Answer to Question
        """
        
    def place_data_in_dict(self, path, labels, text):
        if 'train' in path or 'training' in path:
            key = 'train'
        elif 'dev' in path or 'val' in path or 'validation' in path or 'valid' in path:
            key = 'dev'
        elif 'test' in path:
            key = 'test'
        self.data[key]['labels'] = labels
        self.data[key]['text'] = text


class LSTMProcesses(nn.Module):
    def __init__(self, data_flag, dimension=128):
        super(LSTMProcesses, self).__init__()
        # prepare the label and text fields 
        self.datasets = DataSets()
        self.data_flag = data_flag
        # neural net properties 
        self.embedding = nn.Embedding(len(text_field.vocab), 300)
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=300,
                            hidden_size=dimension,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2*dimension, 1)
    
    def forward(self, text, text_len):
        text_emb = self.embedding(text)
        packed_input = pack_padded_sequence(text_emb, text_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out_forward = output[range(len(output)), text_len - 1, :self.dimension]
        out_reverse = output[:, 0, self.dimension:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)
        text_fea = self.drop(out_reduced)
        text_fea = self.fc(text_fea)
        text_fea = torch.squeeze(text_fea, 1)
        text_out = torch.sigmoid(text_fea)
        return text_out

    def prep_data(self):
        # set up fields
        label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        text_field = Field(tokenize='spacy', lower=True, include_lengths=True, batch_first=True)
        text_field.build_vocab(self.datasets.data['train']['text'], min_freq=3)
        # read in all of the data
        if self.data_flag == 'lambeq_default':
            self.datasets.get_default_lambeq_data()
        elif self.data_flag == 'news_data':
            self.datasets.get_news_data()
        # set up iterators
        for key in self.datasets.data.keys():
            # combine data and labels 
            iter = BucketIterator(self.datasets.data[key]['text'], batch_size=32, sort_key=lambda x: len(x.text),
                            device=device, sort=True, sort_within_batch=True)
            self.datasets.data[key]['iterator'] = iter
        

# run lambeq on default data
# QP = LambeqProcesses(data_flag='lambeq_default')
# QP.main()

# run lambeq on news data 
# QP = LambeqProcesses(data_flag='news_data')
# QP.main()

# run LSTM on default data
# LSTMP = LSTMProcesses(data_flag='lambeq_default')
# LSTMP.main()

# run LSTM on news data
# LSTMP = LSTMProcesses(data_flag='news_data')
# LSTMP.main()
