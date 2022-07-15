import os
import sys
from handle_inputs import get_args

parameters = get_args()

def run_lambeq_on_default():
    # run lambeq on default data
    QP = LambeqProcesses(data_flag='lambeq_default')
    QP.main()


def run_lambeq_on_news():
    # run lambeq on news data
    QP = LambeqProcesses(data_flag='news_data')
    QP.main()


def run_lstm_on_default():
    # run LSTM on default data
    LSTMP = LSTMProcesses(data_flag='lambeq_default')
    LSTMP.main()


def run_lstm_on_news():
    # run LSTM on news data
    LSTMP = LSTMProcesses(data_flag='news_data')
    LSTMP.main()
