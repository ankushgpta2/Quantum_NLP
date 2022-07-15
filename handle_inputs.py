import argparse


def get_args():
    # initialize parser
    parser = argparse.ArgumentParser(description="Parameters For Neural Nets")
    # general arguments and default values
    parser.add_argument('--flag_for_lambeq_default', type=bool, default=True)
    parser.add_argument('--flag_for_lambeq_news', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_default', type=bool, default=False)
    parser.add_argument('--flag_for_lstm_news', type=bool, default=False)
    # get the actual arguments
    args = get_args().parse_args()
    parameters = {'flag_for_lambeq_default': args.flag_for_lambeq_default, 'flag_for_lambeq_news': flag_for_lambeq_news,
                  'flag_for_lstm_default': flag_for_lstm_default, 'flag_for_lstm_news': flag_for_lstm_news}
    return parameters