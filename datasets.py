import numpy as np
import pandas as pd


class DataSets:
    def __init__(self):
        self.data = {'train': {}, 'dev': {}, 'test': {}}

    def get_default_lambeq_data(self):
        """ The default data set from the lambeq GitHub repo (binary-classes)
        """
        # explicitly specify the data
        data_paths = ['datasets/lambeq_default_data/mc_train_data.txt', 'datasets/lambeq_default_data/mc_dev_data.txt',
                      'datasets/lambeq_default_data/mc_test_data.txt']
        for path in data_paths:
            labels, sentences = [], []
            with open(path) as f:
                for line in f:
                    t = float(line[0])
                    labels.append([t, 1-t])
                    sentences.append(line[1:].strip())
            self.place_data_in_dict(path=path, labels=labels, text=sentences)

        self.convert_dict_to_csv(path_for_csv='datasets/lambeq_default_data/mc_full_data.csv')

    def get_news_data(self):
        """ News data with news titles and texts (binary-classes)
        """
        # read the data from .csv file
        data_paths = ['datasets/news_classification_true_false/train.csv',
                      'datasets/news_classification_true_false/valid.csv',
                      'datasets/news_classification_true_false/test.csv']
        for path in data_paths:
            df = pd.read_csv(path)
            df = df.sample(frac=1).reset_index(drop=True)
            self.place_data_in_dict(path=path, text=df['title'][:10].to_list(), labels=df['label'][:10].to_list())

        self.convert_dict_to_csv(path_for_csv='datasets/news_classification_true_false/full.csv')

    def place_data_in_dict(self, path, labels, text):
        """
        Places the texts and labels from dataset into a dictionary for lambeq to properly process it

        Parameters
        ----------
        path : str
            Path to the original location of the dataset
        labels : list
            Labels for the text
        text : list
            Pieces of text

        Returns
        -------
        self.data[train or val or test][labels or text] : dict (side-effect)
            Dictionary containing the texts and labels for each dataset split
        """
        # clean the text
        [string.replace('"', '') for string in text]
        [string.replace("'", "") for string in text]
        if 'train' in path or 'training' in path:
            key = 'train'
        elif 'dev' in path or 'val' in path or 'validation' in path or 'valid' in path:
            key = 'dev'
        elif 'test' in path:
            key = 'test'
        # check if the labels and text are the same size
        if len(labels) != len(text):
            raise ValueError(f'Labels and text must have the same length')
        # random permutation to labels and text (in unison)
        p = np.random.permutation(len(labels))
        labels, text = np.array(labels)[p.astype(int)], np.array(text)[p.astype(int)]
        self.data[key]['labels'] = labels.tolist()
        self.data[key]['text'] = text.tolist()

    def convert_dict_to_csv(self, path_for_csv):
        """
        Converts the self.data dictionary to a CSV at the specified path

        Parameters
        ----------
        path_for_csv : str
            Specified path to save the .CSV file to

        Returns
        -------
        file : .CSV (side-effect)
            A .CSV file is saved to the specified path
        """
        full_text, full_labels = [], []
        for split in self.data.keys():
            full_text.extend(self.data[split]['text'])
            full_labels.extend(self.data[split]['labels'])
        df = pd.DataFrame({
            'text': full_text,
            'labels': full_labels,
        })
        try:
            if len(df['labels'][0]) > 1:
                corrected_labels = [x.index(1.0) for x in df['labels']]
                df['labels'] = corrected_labels
        except:
            pass
        df.to_csv(path_for_csv, index=False)

