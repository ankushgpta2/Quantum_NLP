import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import numpy as np
import datetime
from datetime import date, datetime
# lambeq related packages
from lambeq import BobcatParser, remove_cups, AtomicType, IQPAnsatz, TketModel, QuantumTrainer, SPSAOptimizer, \
    Dataset
from pytket.extensions.qiskit import AerBackend


class LambeqProcesses(object):
    def __init__(self, dataset, data_flag):
        self.reader = BobcatParser(verbose='text')
        self.ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1},
                   n_layers=1, n_single_qubit_params=3)
        self.backend = AerBackend()
        self.model = TketModel
        self.loss_function = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
        self.acc_function = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
        self.epochs = 100
        self.optimizer = SPSAOptimizer
        self.batch_size = 30
        self.dataset = dataset
        self.data_flag = data_flag

    def train(self):
        # use parser and circuit creator
        self.get_diagram_and_circuit()
        # preparation stuff for training
        train_dataset, val_dataset, trainer, model = self.prep_for_train()
        # run the training on generated circuit representations + labels
        trainer.fit(train_dataset, val_dataset, logging_step=12)
        # plot the performance
        self.plot_performance(trainer, model)
        # clear the run logs generated
        os.rmdir('runs')

    # for the parser
    def get_diagram_and_circuit(self):
        for key in self.dataset.keys():
            raw_diagram = self.reader.sentences2diagrams(self.dataset[key]['text'])
            diagram = [remove_cups(diagram) for diagram in raw_diagram]
            circuit = [self.ansatz(sub_diagram) for sub_diagram in diagram]
            # diagram[0].draw()
            # circuit[0].draw()
            self.dataset[key]['diagram'] = diagram
            self.dataset[key]['circuit'] = circuit

    # intializing necessary things for training
    def prep_for_train(self):
        backend_config = {
            'backend': self.backend,
            'compilation': self.backend.default_compilation_pass(2),
            'shots': 8192
        }
        model = self.model.from_diagrams(self.dataset['train']['circuit']+self.dataset['dev']
                        ['circuit']+self.dataset['test']['circuit'], backend_config=backend_config)
        trainer = QuantumTrainer(
            model,
            loss_function=self.loss_function,
            epochs=self.epochs,
            optimizer=self.optimizer,
            optim_hyperparams={'a': 0.05, 'c': 0.06, 'A': 0.01*self.epochs},
            evaluate_functions={'acc': self.acc_function},
            evaluate_on_train=True,
            verbose = 'text',
            seed=0
        )
        train_dataset = Dataset(self.dataset['train']['circuit'], self.dataset['train']['labels'],
                        batch_size=self.batch_size)
        val_dataset = Dataset(self.dataset['dev']['circuit'], self.dataset['dev']['labels'],
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
        # save everything
        if os.path.isdir('figures') is False:
            os.mkdir('figures')
        if os.path.isdir('figures/lambeq') is False:
            os.mkdir('figures/lambeq')
        today = date.today()
        current_date = str(today.strftime("%m/%d/%y")).replace("/", "_")
        now = datetime.now()
        current_time = str(now.strftime("%H:%M:%S"))
        if os.path.isdir('figures/lambeq/' + current_date) is False:
            os.mkdir('figures/lambeq/' + current_date)
        plt.savefig('figures/lambeq/' + current_date + '/loss_acc_' + current_time + '_' + str(self.data_flag) + '.png')
        test_acc = self.acc_function(model(self.dataset['test']['circuit']), self.dataset['test']['labels'])
        print('Test accuracy:', test_acc)
        plt.show()

