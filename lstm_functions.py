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
from datetime import date, datetime
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
        """
        Forward propagation of the LSTM

        Parameters
        ----------
        text : torch(str)
            The piece of text fed into the LSTM for a training step

        Returns
        -------
        logits : torch.Tensor
            Logits or probabilities for the input text belonging to each of the classes
        """
        # embedding the texts... structuring the latent space based on specific input text properties
        embedded = self.embedding(text)
        # running the embedded representation to a dropout layer (for preventing overfitting)
        x = self.dropout(embedded)
        # pass the dropout output to a LSTM
        output, (hidden, cell) = self.lstm(x)
        # get rid of redundant dimension
        hidden.squeeze_(0)
        # pass the output from LSTM to linear activation layer
        logits = self.fc(hidden)
        return logits


class RunLSTM:
    def __init__(self, parameters, data_flag, splits, root_path):
        # passed in variables
        self.embedding_dim = parameters['lstm_embedding_dim']
        self.vocab_size = parameters['lstm_vocab_size']
        self.batch_size = parameters['lstm_batch_size']
        self.hidden_dim = parameters['lstm_hidden_dim']
        self.num_epochs = parameters['lstm_epochs']
        self.lr = parameters['lstm_lr']
        self.data_flag = data_flag
        self.full_csv_path = os.path.join(root_path, 'split_data/full.csv')
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
        self.metrics = {'val_acc': [], 'val_loss': [], 'train_loss': []}
        self.thresholds = np.arange(0.0, 1.01, 0.2)
        # get the date and time
        today = date.today()
        now = datetime.now()
        self.current_date = str(today.strftime("%m_%d_%y"))
        self.current_time = str(now.strftime("%I_%M_%S_%p"))
        self.save_path = os.path.join(root_path, f'figures/lstm/{self.data_flag}/{self.current_date}/{self.current_time}')

    def prep_data(self):
        """
        Tokenizes the .CSV data file and splits it into training/validation/test sets as DataLoaders

        Returns
        -------
        self.text_tokenizer : torchtext.legacy.data.Field (side-effect)
            Tokenizer instance for the texts
        self.label_tokenizer : torchtext.legacy.data.Field (side-effect)
            Tokenizer instance for the labels
        self.train_loader : torch.DataLoader (side-effect)
            Batched training data split
        self.valid_loader : torch.DataLoader (side-effect)
            Batched validation data split
        self.test_loader : torch.DataLoader (side-effect)
            Batched test data split
        """
        # tokenize the words
        self.text_tokenizer = torchtext.legacy.data.Field(
            tokenize='spacy',
            tokenizer_language='en_core_web_sm'
        )
        self.label_tokenizer = torchtext.legacy.data.LabelField(dtype=torch.long)
        fields = [('text', self.text_tokenizer), ('labels', self.label_tokenizer)]

        dataset = torchtext.legacy.data.TabularDataset(
            path=self.full_csv_path, format='csv',
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

    def run_lstm(self):
        """ Main function for running through the training and eval steps + saving best model and plots
        """
        self.prep_data()
        class_list = []
        [class_list.extend(batch.labels.cpu().detach().numpy()) for batch in self.train_loader]
        output_dim = len(np.unique(np.array(class_list)))
        # for the model itself
        model = LSTMProcesses(input_dim=len(self.text_tokenizer.vocab),
                              embedding_dim=self.embedding_dim,
                              hidden_dim=self.hidden_dim,
                              output_dim=output_dim,
                              )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        model = model.to(self.device)
        self.train(model=model, optimizer=optimizer)
        test_acc, test_loss = self.eval(model=self.best_model, data_loader=self.test_loader)
        print(f'\nFor Best Model, Test Accuracy = {np.round(test_acc, 1)} and Test Loss = {np.round(test_loss, 5)}')
        self.save_best_model()
        self.plot_results()
        self.save_metadata_to_text()

    def train(self, model, optimizer):
        """
        Main training loop

        Parameters
        ----------
        model : torch.nn.model
            Pytorch LSTM model
        optimizer : torch.nn.optimizer
            Optimizer for updating weights based on calculated loss gradient
        """
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
            self.metrics['train_loss'].append(avg_loss)
            # perform validation after each epoch
            with torch.set_grad_enabled(False):
                accuracy, val_loss = self.eval(model=model, data_loader=self.valid_loader)
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_epoch = epoch
                    self.best_model = copy.deepcopy(model)
                print(f'{epoch}/{self.num_epochs} ----> train loss = {np.round(avg_loss, 4)} ----> val loss = '
                      f'{np.round(val_loss, 4)} ----> accuracy = {np.round(accuracy, 2)}')

    def eval(self, model, data_loader):
        """
        Evaluation --> both validation and final test on valid_laoder and test_loader respectively

        Parameters
        ----------
        model : torch.nn.model
            Pytorch LSTM model
        data_loader : torch.DataLoader
            Either the valid_loader for validation or test_loader for final test

        Returns
        -------
        accuracy : float
            Correctly predicted labels / total number of labels
        loss : float
            Cross entropy loss between logits and labels
        """
        with torch.no_grad():
            correct_pred, total, avg_loss = 0, 0, 0
            accuracy_holder, val_loss_holder = [], []
            for i, (text, labels) in enumerate(data_loader):
                text, labels = text.to(self.device), labels.float().to(self.device)
                logits = model(text)
                _, predicted_labels = torch.max(logits, 1)
                total += labels.size(0)
                correct_pred = (predicted_labels == labels).sum()
                loss = F.cross_entropy(logits, labels.long())
                avg_loss += loss.item()
                accuracy_holder.append((correct_pred.float()/total*100).numpy())
        self.metrics['val_acc'].append(np.average(accuracy_holder))
        self.metrics['val_loss'].append(avg_loss)
        return np.average(accuracy_holder), avg_loss

    def plot_results(self):
        """
        Plotting the final results from training and validation and saving it locally

        Returns
        -------
        plot : matplotlib.figure (side-effect)
            Figure saved locally containing the loss and accuracy for training and validation datasets
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey='row', figsize=(10, 6))
        ax1.plot(self.metrics['train_loss'], color='blue')
        ax2.plot(self.metrics['val_loss'], color='red')
        ax3.plot(self.metrics['val_acc'], color='green')
        ax3.set_xlabel('Epochs'), ax1.set_ylabel('Train Loss'), ax2.set_ylabel('Val Loss'), ax3.set_ylabel('Accuracy')
        plt.suptitle('Loss and Accuracy Performance with LSTM', fontweight='bold', fontsize=14)
        # get the saving path and check if the directories exist
        save_path_split = list(self.save_path.split('/'))
        for i in range(1, len(save_path_split)+1):
            directory = "/".join([save_path_split[x] for x in range(i)])
            if os.path.isdir(directory) is False:
                os.mkdir(directory)
                os.chmod(directory, stat.S_IRWXO)
        # save the figure
        plt.savefig(f'{self.save_path}/loss_acc_figures.png')
        # plt.show()

    def save_best_model(self):
        """ Saves the best performing model (based on validation performance, accuracy by default)
        """
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

    def save_metadata_to_text(self):
        with open(os.path.join(self.save_path, 'metadata.txt'), 'w') as f:
            f.write(f'embedding dim = {self.embedding_dim}\n'
                    f'vocab size = {self.vocab_size}\n'
                    f'batch size = {self.batch_size}\n'
                    f'hidden dim = {self.hidden_dim}\n'
                    f'num epochs = {self.num_epochs}\n'
                    f'lr = {self.lr}\n'
                    f'data flag = {self.data_flag}\n'
                    f'splits = {self.train_split}, {self.val_split}, {self.test_split}\n'
                    f'date = {self.current_date}\n'
                    f'time = {self.current_time}\n'
                    )

    def predict_sentiment(self, model, sentence):
        model.eval()
        nlp = spacy.blank('en')
        tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
        indexed = [self.text_tokenizer.vocab.stoi[t] for t in tokenized]
        tensor = (torch.LongTensor(indexed).to(self.device)).unsqueeze(1)
        prediction = torch.nn.functional.softmax(model(tensor), dim=1)
        return prediction[0][0].item()
