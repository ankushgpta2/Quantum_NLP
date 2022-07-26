# torch packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.legacy.data.field
import pandas as pd
import numpy as np
import spacy


class LSTMProcesses(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTMProcesses, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        # neural net properties
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden.squeeze_(0)
        output = self.fc(hidden)
        return output


class RunLSTM:
    def __init__(self, data_flag, embedding_dim, context_size, vocab_size, batch_size, hidden_dim, num_classes, num_epochs):
        # passed in variables
        self.data_flag = data_flag
        self.embedding_dim = embedding_dim
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        # self.L = LSTMProcesses(context_size=self.context_size, embedding_dim=self.embedding_dim, vocab_size=20000)
        # just instantiate a few other variables
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.label_tokenizer = None
        self.text_tokenizer = None

    def prep_data(self):
        df = pd.read_csv('datasets/mc_full_data.csv')

        # tokenize the words
        self.text_tokenizer = torchtext.legacy.data.Field(
            tokenize='spacy',
            tokenizer_language='en_core_web_sm'
        )
        self.label_tokenizer = torchtext.legacy.data.LabelField(dtype=torch.long)
        fields = [('text', self.text_tokenizer), ('labels', self.label_tokenizer)]
        if self.data_flag == 'lambeq_default_data':
            path = 'datasets/mc_full_data.csv'
        else:
            # TODO: need to double check that the fields above captures column names properly in this CV
            path = 'datasets/news_classification_true_false/train.csv'

        dataset = torchtext.legacy.data.TabularDataset(
            path=path, format='csv',
            skip_header=True, fields=fields
        )

        # now split the data
        train_data, test_data = dataset.split(
            split_ratio=[0.8, 0.2],
        )
        train_data, valid_data = train_data.split(
            split_ratio=[0.85, 0.15],
        )

        # build the necessary vocab (just to see)
        self.text_tokenizer.build_vocab(train_data, max_size=self.vocab_size)
        self.label_tokenizer.build_vocab(train_data)

        self.train_loader, self.valid_loader, self.test_loader = torchtext.legacy.data.BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=self.batch_size,
            device=self.device,
            sort_within_batch=False,
            sort_key=lambda x: len(x.text)
        )

    def train(self):
        self.prep_data()
        # for the model itself
        model = LSTMProcesses(input_dim=len(self.text_tokenizer.vocab),
                                   embedding_dim=self.embedding_dim,
                                   hidden_dim=self.hidden_dim,
                                   output_dim=self.num_classes
                              )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model = model.to(self.device)

        for epoch in range(1, self.num_epochs+1):
            model.train()
            avg_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                text, labels = batch.text.to(self.device), batch.labels.to(self.device)
                logits = model(text)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
            # perform validation after each epoch
            with torch.set_grad_enabled(False):
                accuracy = self.validation(model=model)
            print(str(epoch) + '/' + str(self.num_epochs) + ' ----> train loss = ' + str(np.round(avg_loss, 9))
                  + ' ----> accuracy = ' + str(accuracy))

    def validation(self, model):
        with torch.no_grad():
            correct_val_pred, total = 0, 0
            for i, (val_text, val_labels) in enumerate(self.valid_loader):
                val_text, val_labels = val_text.to(self.device), val_labels.float().to(self.device)
                val_logits = model(val_text)
                _, predicted_labels = torch.max(val_logits, 1)
                total += val_labels.size(0)
                correct_val_pred = (predicted_labels == val_labels).sum()
        return (correct_val_pred.float()/total*100).numpy()

    def predict_sentiment(self, model, sentence):
        model.eval()
        nlp = spacy.blank('en')
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [self.text_tokenizer.vocab.stoi[t] for t in tokenized]
        tensor = (torch.LongTensor(indexed).to(self.device)).unsqueeze(1)
        prediction = torch.nn.functional.softmax(model(tensor), dim=1)
        return prediction[0][0].item()

