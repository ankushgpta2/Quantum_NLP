import os
import sys
# torch packages
import torch
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
        self.fc = nn.Linear( 2 *dimension, 1)

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
