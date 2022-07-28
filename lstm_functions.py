import os
import stat
import copy
import matplotlib.pyplot as plt
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext.legacy.data.field
from pathlib import Path


class LSTMProcesses(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(LSTMProcesses, self).__init__()
        # neural net properties
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        embedded = self.embedding(text)
        x = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(x)
        hidden.squeeze_(0)
        logits = self.fc(hidden)
        return logits


class RunLSTM:
    def __init__(self, parameters, data_flag, splits):
        # passed in variables
        self.embedding_dim = parameters['lstm_embedding_dim']
        self.vocab_size = parameters['lstm_vocab_size']
        self.batch_size = parameters['lstm_batch_size']
        self.hidden_dim = parameters['lstm_hidden_dim']
        self.num_classes = parameters['lstm_num_classes']
        self.num_epochs = parameters['lstm_epochs']
        self.lr = parameters['lstm_lr']
        self.data_flag = data_flag
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_split = splits['train']
        self.val_split = splits['val']
        self.test_split = splits['test']
        # just instantiate a few other variables
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.label_tokenizer = None
        self.text_tokenizer = None
        self.best_model = None
        self.best_acc = 0.0
        self.best_epoch = None
        self.train_loss_holder, self.val_acc_holder, self.val_loss_holder = [], [], []

    def prep_data(self):
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
            path = 'datasets/news_classification_true_false/full.csv'

        dataset = torchtext.legacy.data.TabularDataset(
            path=path, format='csv',
            skip_header=True, fields=fields
        )

        # now split the data
        train_data, test_data = dataset.split(
            split_ratio=[1-self.test_split, self.test_split],
        )
        train_data, valid_data = train_data.split(
            split_ratio=[self.train_split, self.val_split],
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
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
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
            self.train_loss_holder.append(avg_loss)
            # perform validation after each epoch
            with torch.set_grad_enabled(False):
                accuracy, val_loss = self.eval(model=model, data_loader=self.valid_loader)
                self.val_acc_holder.append(accuracy)
                self.val_loss_holder.append(val_loss)
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_epoch = epoch
                    self.best_model = copy.deepcopy(model)
                print(f'{epoch}/{self.num_epochs} ----> train loss = {np.round(avg_loss, 9)} ----> val loss = {np.round(val_loss, 9)} '
                  f'---> accuracy = {np.round(accuracy, 2)}')
        test_acc, test_loss = self.eval(model=self.best_model, data_loader=self.test_loader)
        print(f'\nFor Best Model, Test Accuracy = {np.round(test_acc, 1)} and Test Loss = {np.round(test_loss, 5)}')
        self.save_best_model()
        self.plot_results()

    def eval(self, model, data_loader):
        with torch.no_grad():
            correct_pred, total, avg_loss = 0, 0, 0
            for i, (text, labels) in enumerate(data_loader):
                text, labels = text.to(self.device), labels.float().to(self.device)
                logits = model(text)
                _, predicted_labels = torch.max(logits, 1)
                total += labels.size(0)
                correct_pred = (predicted_labels == labels).sum()
                loss = F.cross_entropy(logits, labels.long())
                avg_loss += loss.item()
        return (correct_pred.float()/total*100).numpy(), avg_loss

    def plot_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey='row', figsize=(10, 6))
        ax1.plot(self.train_loss_holder, color='blue')
        ax2.plot(self.val_loss_holder, color='red')
        ax3.plot(self.val_acc_holder, color='green')
        ax3.set_xlabel('Epochs'), ax1.set_ylabel('Train Loss'), ax2.set_ylabel('Val Loss'), ax3.set_ylabel('Accuracy')
        plt.suptitle('Loss and Accuracy Performance with LSTM', fontweight='bold', fontsize=14)
        plt.show()

    def save_best_model(self):
        name_of_dir = 'best_model'
        path_for_file = Path(f'{name_of_dir}/model.pt')
        if os.path.isdir(name_of_dir) is False:
            os.mkdir(name_of_dir)
            os.chmod(name_of_dir, stat.S_IRWXO)
        torch.save({'acc': self.best_acc, 'epoch': self.best_epoch, 'model': self.best_model}, path_for_file)
        try:
            assert torch.load(path_for_file)
        except:
            raise ValueError(f'Saved .pt file for best model in {path_for_file} cannot be read back in. Please check.')

    def predict_sentiment(self, model, sentence):
        model.eval()
        nlp = spacy.blank('en')
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [self.text_tokenizer.vocab.stoi[t] for t in tokenized]
        tensor = (torch.LongTensor(indexed).to(self.device)).unsqueeze(1)
        prediction = torch.nn.functional.softmax(model(tensor), dim=1)
        return prediction[0][0].item()
