import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'True'
import datetime
import matplotlib.pyplot as plt
import numpy as np
import shutil
from datetime import date, datetime
from lambeq import BobcatParser, remove_cups, AtomicType, IQPAnsatz, TketModel, QuantumTrainer, SPSAOptimizer, \
    Dataset
from pytket.extensions.qiskit import AerBackend


class LambeqProcesses(object):
    def __init__(self, dataset, data_flag, parameters):
        self.reader = BobcatParser(verbose='text')
        self.ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=1, n_single_qubit_params=3)
        self.backend = AerBackend()
        self.model = TketModel
        self.loss_function = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
        self.acc_function = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
        self.num_epochs = parameters['lambeq_epochs']
        self.optimizer = SPSAOptimizer
        self.batch_size = parameters['lambeq_batch_size']
        self.dataset = dataset
        self.data_flag = data_flag
        # get the date and time
        today = date.today()
        now = datetime.now()
        self.current_date = str(today.strftime("%m_%d_%y"))
        self.current_time = str(now.strftime("%I_%M_%S_%p"))

    def train(self):
        """ Main function for running the training process and plotting of the train/eval results
        """
        # use parser and circuit creator
        train_dataset, val_dataset = self.get_diagram_and_circuit()
        # preparation stuff for training
        trainer = self.prep_for_train()
        # run the training on generated circuit representations + labels
        trainer.fit(train_dataset, val_dataset, logging_step=1)
        # plot the performance
        self.plot_performance(trainer, self.model)
        # save metadata
        self.save_metadata_to_text()
        # clear the run logs generated
        shutil.rmtree('runs', ignore_errors=False, onerror=None)

    def get_diagram_and_circuit(self):
        """
        Function for generating the tree diagram of the inputted text and translating it to a quantum representation

        Returns
        -------
        train_dataset : lambeq.Dataset
            Batched dataset based on the circuit representation of the training text
        val_dataset : lambeq.Dataset
            Batched dataset based on the circuit representation of the validation text
        """
        for key in ['train', 'dev', 'test']:
            # generate the raw diagram tree for all the text
            raw_diagram = self.reader.sentences2diagrams(self.dataset[key]['text'])
            # removing the cups from the diagram
            diagram = [remove_cups(diagram) for diagram in raw_diagram]
            # quantum representation or circuit constructed from the tree diagram
            circuit = [self.ansatz(sub_diagram) for sub_diagram in diagram]
            # diagram[0].draw()
            # circuit[0].draw()
            # save the diagram and circuit representations of the text under separate keys within self.dataset dict
            self.dataset[key]['diagram'] = diagram
            self.dataset[key]['circuit'] = circuit
        # get the training dataset and validation dataset
        train_dataset = Dataset(self.dataset['train']['circuit'], self.dataset['train']['labels'],
                                batch_size=self.batch_size)
        val_dataset = Dataset(self.dataset['dev']['circuit'], self.dataset['dev']['labels'],
                                batch_size=self.batch_size)
        return train_dataset, val_dataset

    def prep_for_train(self):
        """
        Function for initializing the quantum trainer model and train/val datasets

        Returns
        -------
        trainer : lambeq.QuantumTrainer
            Actual model for training, utilizing specified backend configuration and other hyperparameters (epochs,
            loss function, optimizer, and evaluate function)
        """
        # set up the backend configurator for the model
        backend_config = {
            'backend': self.backend,
            'compilation': self.backend.default_compilation_pass(2),
            'shots': 8192
        }
        # set up the final model using TketModel
        self.model = self.model.from_diagrams(self.dataset['train']['circuit']+self.dataset['dev']
                        ['circuit']+self.dataset['test']['circuit'], backend_config=backend_config)
        # set up the final quantum trainer, using the model initialized above, with necessary hyperparameters
        trainer = QuantumTrainer(
            self.model,
            loss_function=self.loss_function,
            epochs=self.num_epochs,
            optimizer=self.optimizer,
            optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.01*self.num_epochs},
            evaluate_functions={'acc': self.acc_function},
            evaluate_on_train=True,
            verbose='text',
            seed=0
        )
        return trainer

    def plot_performance(self, trainer, model):
        """
        Plots the loss and accuracy of the lambeq model on the training and development datasets

        Parameters
        ----------
        trainer : lambeq.QuantumTrainer object
            The quantum trainer fitted on the training and validation datasets, containing accuracy and loss information
        model : TketModel object
            Actual model structure used by the quantum trainer to fit on the train/val datasets

        Returns
        -------
        plot : matplotlib.figure (side-effect)
            Plot figure is saved at /figures/lambeq/{current_date}/loss_acc_{current_time}_lambeq.png containing the
            loss and accuracy information on the train/dev datasets
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
        ax1.plot(trainer.train_epoch_costs, color='blue')
        ax2.plot(trainer.val_costs, color='red')
        ax3.plot(trainer.train_results['acc'], color='blue')
        ax4.plot(trainer.val_results['acc'], color='red')
        # labels
        ax1.set_title('Training Set'), ax2.set_title('Development Set'), ax3.set_xlabel('Epochs'),
        ax4.set_xlabel('Epochs'), ax1.set_ylabel('Loss'), ax3.set_ylabel('Accuracy')
        plt.suptitle('Loss and Accuracy Performance on Train/Dev Sets', fontweight='bold', fontsize=14)
        # get the saving path and check if the directories exist
        save_path = f'figures/lambeq/{self.current_date}/{self.current_time}'
        save_path_split = list(save_path.split('/'))
        for i in range(1, len(save_path_split) + 1):
            directory = "/".join([save_path_split[x] for x in range(i)])
            if os.path.isdir(directory) is False:
                os.mkdir(directory)
        # save the figure
        plt.savefig(f'{save_path}/{self.data_flag}.png')
        test_acc = self.acc_function(model(self.dataset['test']['circuit']), self.dataset['test']['labels'])
        print('Test accuracy:', test_acc)
        # plt.show()

    def save_metadata_to_text(self):
        save_path = f'figures/lambeq/{self.current_date}/{self.current_time}'
        with open(os.path.join(save_path, 'metadata.txt'), 'w') as f:
            f.write(f'batch size = {self.batch_size}\n'
                    f'num epochs = {self.num_epochs}\n'
                    f'data flag = {self.data_flag}\n'
                    f'date = {self.current_date}\n'
                    f'time = {self.current_time}\n'
                    )

    def plot_confusion_matrix(self):
        """
        :return:
        """

    def plot_roc_curve(self):
        """
        :return:
        """

