# torch packages
import torch
import torch.nn as nn
from torch import autograd, optim
import torch.nn.functional as F
from datasets import *
import numpy as np


class LSTMProcesses(nn.Module):
    def __init__(self, context_size, embedding_dim, vocab_size):
        super(LSTMProcesses, self).__init__()
        # prepare the label and text fields
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        # neural net properties
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear1 = nn.Linear(context_size * self.embedding_dim, 128)
        self.linear2 = nn.Linear(128, self.vocab_size)
        self.lstm = nn.LSTM(embedding_dim, 2, bidirectional=True)

    def forward(self, input):
        embeds = self.word_embeddings(input).view((1, -1))
        lstm_out, _ = self.lstm(embeds.view(len(input), 1, -1))
        # out = F.relu(self.linear1(embeds))
        # out = self.linear2(out)
        log_probs = F.log_softmax(lstm_out, dim=1)
        print(np.shape(log_probs))
        return log_probs


class RunLSTM:
    def __init__(self, dataset, embedding_dim, context_size):
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.context_size = context_size

    @staticmethod
    def convert_to_wordidx(sentence):
        vocab = set(sentence)
        print(vocab)
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        return word_to_ix, vocab

    @staticmethod
    def get_ngram(sentence):
        CONTEXT_SIZE = 2
        ngrams = [
            (
                [sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
                sentence[i]
            )
            for i in range(CONTEXT_SIZE, len(sentence))
        ]
        return ngrams

    def main(self):
        sentence = self.dataset['train']['text'][0]
        word_to_ix, vocab = self.convert_to_wordidx(sentence=sentence.split())
        ngrams = self.get_ngram(sentence=sentence.split())
        model = LSTMProcesses(context_size=self.context_size, embedding_dim=self.embedding_dim, vocab_size=len(vocab))
        losses = []
        loss_function = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)
        for epoch in range(1, 100+1):
            total_loss = 0
            for context, target in ngrams:
                context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
                model.zero_grad()
                log_probs = model(context_idxs)
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(str(epoch) + '/' + str(100) + ':  ' + str(total_loss))
            losses.append(total_loss)
        print(losses)
