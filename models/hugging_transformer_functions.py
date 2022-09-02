from transformers import DistilBertConfig, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, \
    Trainer, DataCollatorWithPadding, DistilBertForSequenceClassification
import torch.utils.data.dataset
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import os
import matplotlib.pyplot as plt
import stat
from datetime import date, datetime


class HuggingBertModel(nn.Module):
    def __init__(self, dataset, use_pretrained, parameters, root_path, data_flag):
        super(HuggingBertModel, self).__init__()

        # get hyperparameters for neural network
        self.parameters = parameters
        self.batch_size = parameters['transformer_batch_size']
        self.num_epochs = parameters['transformer_epochs']
        self.lr = parameters['transformer_lr']
        self.weight_decay = parameters['transformer_weight_decay']
        self.metric_for_best_model = parameters['transformer_metric_for_best_model']

        # get the dataset
        self.data_flag = data_flag
        self.dataset = dataset

        # get the directories
        self.root_path = root_path
        today = date.today()
        now = datetime.now()
        self.current_date = str(today.strftime("%m_%d_%y"))
        self.current_time = str(now.strftime("%I_%M_%S_%p"))
        self.save_path = os.path.join(root_path, f'results/transformer/{data_flag}/{self.current_date}/{self.current_time}')
        self.path_to_figures = os.path.join(self.save_path, 'figures')
        self.path_to_best_model = os.path.join(self.save_path, 'best_model')

        # initialize necessary pre-processing models
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # initialize configuration for the model BERT model and pass in args for specific dataset
        self.config = self.get_mod_config()

        # load it into the model
        if use_pretrained is True:
            self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                        num_labels=len(set(self.dataset['train']['labels'].tolist())))
        else:
            self.model = DistilBertForSequenceClassification(self.config)

        # tokenize the input text and convert to torch
        self.tokenized_train = self.tokenizer(self.dataset['train']['text'],
                                              truncation=True,
                                              padding=True,
                                              return_tensors='pt',
                                              )
        self.tokenized_val = self.tokenizer(self.dataset['dev']['text'],
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt',
                                            )
        self.train_dataset = TorchedDataset(self.tokenized_train, self.dataset['train']['labels'])
        self.val_dataset = TorchedDataset(self.tokenized_val, self.dataset['dev']['labels'])

        # get the number of steps per epoch
        if ((len(self.train_dataset) / self.batch_size) - (len(self.train_dataset) // self.batch_size)) != 0.0:
            self.steps_per_epoch = (len(self.train_dataset) // self.batch_size) + 1
        else:
            self.steps_per_epoch = len(self.train_dataset) / self.batch_size

        # initialize the training arguments
        self.training_args = TrainingArguments(
            output_dir='/results',
            learning_rate=self.lr,
            num_train_epochs=self.num_epochs,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            weight_decay=self.weight_decay,
            metric_for_best_model=self.metric_for_best_model,
            logging_dir='/results',
            logging_steps=self.steps_per_epoch,
            load_best_model_at_end=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
        )

        # load everything into the Trainer
        self.metrics = {'train_loss': [], 'eval_loss': [], 'eval_acc': []}
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    def run_hugging_bert(self):
        # call train for Trainer
        self.trainer.train()
        # get the log info
        for x in [x for x in self.trainer.state.log_history if 'loss' in x.keys() or 'eval_loss' in x.keys()]:
            try:
                self.metrics['train_loss'].append(x['loss'])
            except:
                self.metrics['eval_loss'].append(x['eval_loss'])
                self.metrics['eval_acc'].append(x['eval_accuracy'])
        # save the best model
        self.trainer.save_model(self.path_to_best_model)
        # plot the results
        self.plot_results()
        # save metadata to json file
        self.save_metadata_to_json()

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def plot_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey='row', figsize=(10, 6))
        ax1.plot(self.metrics['train_loss'], color='blue')
        ax2.plot(self.metrics['eval_loss'], color='red')
        ax3.plot(self.metrics['eval_acc'], color='green')
        ax3.set_xlabel('Epochs'), ax1.set_ylabel('Train Loss'), ax2.set_ylabel('Val Loss'), ax3.set_ylabel('Accuracy')
        plt.suptitle('Loss and Accuracy Performance with Distil-BERT Transformer', fontweight='bold', fontsize=14)
        # get the saving path and check if the directories exist
        save_path_split = list(self.path_to_figures.split('/'))
        for i in range(1, len(save_path_split) + 1):
            directory = "/".join([save_path_split[x] for x in range(i)])
            if os.path.isdir(directory) is False:
                os.mkdir(directory)
                os.chmod(directory, stat.S_IRWXO)
        # save the figure
        plt.savefig(f'{self.path_to_figures}/loss_acc_figures.png')
        # plt.show()

    def save_metadata_to_json(self):
        config_dict = {
            'neural_hyperparams': {
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'metric_for_best_model': self.metric_for_best_model,
                'weight_decay': self.weight_decay,
                'learning_rate': self.lr,
                'steps_per_epoch': self.steps_per_epoch,
            },
            'metadata': {
                'dataset': self.data_flag,
                'splits': self.parameters['transformer_splits'],
                'model': 'transformer',
                'date': self.current_date,
                'time': self.current_time,
            }
        }
        config_file = open(os.path.join(self.save_path, 'meta.json'), "w")
        json.dump(config_dict, config_file, indent=6)
        config_file.close()

    def get_mod_config(self):
        config = DistilBertConfig()
        # TODO: do necessary modifications to BERT for dataset and task... change hidden_dim to number of labels
        return config


class TorchedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

