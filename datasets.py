import pandas as pd


class DataSets:
    def __init__(self):
        self.data = {'train': {}, 'dev': {}, 'test': {}}

    def get_default_lambeq_data(self):
        """ Text Classification
        """
        # explicitly specify the data
        data_paths = ['datasets/mc_train_data.txt', 'datasets/mc_dev_data.txt', 'datasets/mc_test_data.txt']
        for path in data_paths:
            labels, sentences = [], []
            with open(path) as f:
                for line in f:
                    t = float(line[0])
                    labels.append([t, 1- t])
                    sentences.append(line[1:].strip())
            self.place_data_in_dict(path=path, labels=labels, text=sentences)

    def get_news_data(self):
        """ Text Classification
        """
        # read the data from .csv file
        data_paths = ['datasets/news_classification_true_false/train.csv',
                      'datasets/news_classification_true_false/valid.csv',
                      'datasets/news_classification_true_false/test.csv']
        for path in data_paths:
            df = pd.read_csv(path)
            df = df.sample(frac=1).reset_index(drop=True)
            self.place_data_in_dict(path=path, text=df['title'][:10].to_list(), labels=df['label'][:10].to_list())

    def place_data_in_dict(self, path, labels, text):
        # clean the text
        [string.replace('"', '') for string in text]
        [string.replace("'", "") for string in text]
        # for i, string in enumerate(text):
            # text[i] = "'" + string + "'"
        if 'train' in path or 'training' in path:
            key = 'train'
        elif 'dev' in path or 'val' in path or 'validation' in path or 'valid' in path:
            key = 'dev'
        elif 'test' in path:
            key = 'test'
        self.data[key]['labels'] = labels
        self.data[key]['text'] = text
