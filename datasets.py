import glob
import os
import stat
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split


class DataSets:
    def __init__(self, splits, model_type, dataset_name, path):
        # initializing the dictionary structure for storing training/validation/test sets
        self.data = {'full': {}, 'train': {}, 'dev': {}, 'test': {}}
        # initializing the wanted splits for training/validation/test (based on the lstm splits, so that lambeq is same)
        self.splits = splits
        # initialize the location where split files would go
        self.splits_location = None
        # model type
        self.model_type = model_type
        self.train_file_names = ['train', 'training', 'Train', 'Training']
        self.valid_file_names = ['dev', 'val', 'validation', 'valid']
        self.full_file_names = ['full', 'total', 'pooled', 'complete', 'main', 'central']
        self.dataset_name = dataset_name
        self.path = path
        self.path_to_original = os.path.join(self.path, 'original_data')
        self.path_to_split = os.path.join(self.path, 'split_data')
        self.expected_files = ['train', 'dev', 'test', 'full']
        self.num_files_in_original = len(os.listdir(self.path_to_original))

    def get_dataset(self):
        if self.dataset_name == 'news':
            self.general_retrieval()

        if self.dataset_name == 'default':
            self.general_retrieval()

        if self.dataset_name == 'corona':
            self.general_retrieval(text_col_name='OriginalTweet', label_col_name='Sentiment')

        if self.dataset_name == 'ecommerce':
            self.general_retrieval()

        if self.dataset_name == 'spam':
            self.general_retrieval()

    def general_retrieval(self, label_col_name='', text_col_name=''):
        self.handle_split_data_dir()
        file_list = self.get_relevant_files()
        for file in file_list:
            df = self.convert_to_df(path_to_file=os.path.join(self.path_to_original, file))

            # check if the .csv file only has text and labels cols
            if len(df.columns) > 2:
                print(f'\n********** CONVERTING DF WITH >2 COLS TO ONLY TEXT AND LABEL COLS **********\n')
                df = df[[text_col_name, label_col_name]]

            # check if the cols are in proper order within .csv file
            if sum([len(str(x)) for x in df[df.columns[0]]]) < sum([len(str(x)) for x in df[df.columns[1]]]):
                print(f'\n********** CHANGING DF COLS TO PROPER ORDER **********\n')
                df.columns = ['labels', 'text']
                df = df[[df.columns[1], df.columns[0]]]

            # change the names of the columns
            df.columns = ['text', 'labels']

            # check if the labels column contains nums or words
            if all(str(label).isdigit() for label in df['labels'].tolist()) is False and self.model_type != 'lambeq':
                print(f'\n********** CONVERTING CATEGORICAL LABELS TO NUM **********\n')
                df = self.convert_text_label_to_num(df, label_col_name=df.columns[-1], text_col_name=df.columns[0])

            split_type = self.get_split_type(file=file)
            self.save_to_dict(text=df['text'].tolist(), labels=df['labels'].tolist(), split_type=split_type)
            df.to_csv(os.path.join(self.path_to_split, f'{split_type}.csv'), index=False)
            self.expected_files.remove(split_type)

        self.check_splits()

    @staticmethod
    def convert_text_label_to_num(df, label_col_name, text_col_name):
        dictionary = dict.fromkeys([z for z in df[label_col_name].unique()], None)
        raw_labels = [x for x in range(len(dictionary.keys()))]
        for key, value in zip(dictionary.keys(), raw_labels):
            dictionary[key] = value
        new_df_dict = {'text': df[text_col_name], 'labels': [dictionary[x] for x in df[label_col_name]]}
        new_df = pd.DataFrame(new_df_dict)
        assert len(new_df['labels'].unique()) == len(df[label_col_name].unique())
        try:
            assert sum(np.isnan(new_df['labels'])) == 0
        except ValueError:
            raise ValueError('List of labels contains a NaN value... check the conversion from categorical label to '
                             'num function on df')
        return new_df

    def get_relevant_files(self):
        file_list = []
        for file in [x for x in os.listdir(self.path_to_original)]:
            os.chmod(os.path.join(self.path_to_original, file), 0o0777)
            conditions_for_file = [os.path.isfile(os.path.join(self.path_to_original, file)),
                                   os.path.splitext(file)[-1] == '.csv',
                                   os.path.splitext(file)[-1] == '.txt',
                                   ]
            if conditions_for_file == [True, False, True] or conditions_for_file == [True, True, False]:
                file_list.append(file)
        return file_list

    def handle_split_data_dir(self):
        if os.path.isdir(os.path.join(self.path, 'split_data')) is True:
            shutil.rmtree((os.path.join(self.path, 'split_data')), ignore_errors=True)
        os.mkdir((os.path.join(self.path, 'split_data')))
        os.chmod((os.path.join(self.path, 'split_data')), stat.S_IWRITE)

    def check_splits(self):
        self.splits_location = os.path.join(self.path, 'split_data')

        # if only the 'full' dataset is present
        if self.expected_files == ['train', 'dev', 'test'] or self.num_files_in_original == 1:
            df = self.convert_to_df(path_to_file=os.path.join(self.splits_location, 'full.csv'))
            # shuffle and split into train and test/val sets
            train, val_test = train_test_split(df, test_size=1 - self.splits['train'])
            val, test = train_test_split(val_test, test_size=self.splits['test']/self.splits['val'])
            # save to dictionary for lambeq
            split_dict = {'train': train, 'dev': val, 'test': test}
            self.save_missing_splits(split_dict=split_dict)

        # if only the 'full' dataset is missing
        elif self.expected_files == ['full']:
            self.generate_full_csv()

        # if only a training and test dataset are present
        elif self.expected_files == ['dev', 'full']:
            df = self.convert_to_df(path_to_file=os.path.join(self.splits_location, 'test.csv'))
            # shuffle and split into test and validation sets
            val, test = train_test_split(df, test_size=self.splits['test']/self.splits['val'])
            # save to dictionary for lambeq
            split_dict = {'dev': val, 'test': test}
            self.save_missing_splits(split_dict=split_dict)
            self.generate_full_csv()

        # check if the correct # of files are in the split directory for dataset
        try:
            assert (len(glob.glob(os.path.join(self.splits_location, '*.csv')))) == 4
        except AssertionError:
            raise AssertionError('Correct # of .csv files are not inside of the splits directory for dataset...'
                                 f' assumes files have certain key words in them: '
                                 f'{self.train_file_names + self.valid_file_names + self.full_file_names}'
                                 )

    def generate_full_csv(self):
        # create a single csv representing the full dataset
        paths_to_csvs = [os.path.join(self.splits_location, file) for file in os.listdir(self.splits_location)]
        full_df = pd.concat(map(pd.read_csv, paths_to_csvs), ignore_index=True)
        full_df.to_csv(os.path.join(self.splits_location, 'full.csv'), index=False)

    def save_missing_splits(self, split_dict):
        for x, y in zip(split_dict.keys(), split_dict.values()):
            self.save_to_dict(split_type=x, labels=y['labels'].tolist(), text=y['text'].tolist())
            y.to_csv(os.path.join(self.splits_location, f'{x}.csv'), index=False)

    def convert_to_df(self, path_to_file):
        if path_to_file.split('.')[1] == 'txt':
            data_from_txt = {'text': [], 'labels': []}
            with open(path_to_file, 'r') as f:
                for line in f:
                    t = float(line[0])
                    label = [t, 1 - t]
                    if self.model_type == 'lstm' and len(label) != 1:
                        label = label.index(1.0)
                    data_from_txt['labels'].append(label)
                    data_from_txt['text'].append(line[1:].strip())
            df = pd.DataFrame(data_from_txt)
        elif path_to_file.split('.')[1] == 'csv':
            df = pd.read_csv(path_to_file, encoding='latin-1')
        else:
            raise ValueError("Assumes either .txt. or .csv file for the dataset format")
        # permutation
        df = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
        return df

    def save_to_dict(self, text, labels, split_type):
        self.data[split_type]['labels'], self.data[split_type]['text'] = labels, text

    def get_split_type(self, file):
        if True in [x in file for x in self.train_file_names]:
            split_type = 'train'
        elif True in [x in file for x in self.valid_file_names]:
            split_type = 'dev'
        elif 'test' in file:
            split_type = 'test'
        elif True in [x in file for x in self.full_file_names]:
            split_type = 'full'
        else:
            if self.num_files_in_original == 1:
                # assume that the single file in the original_data directory is the full dataset
                split_type = 'full'
            else:
                raise NameError(f'Could not identify which split that file belongs to... assumes file '
                                              f'name has one of the following words in it: '
                                              f'{self.train_file_names + self.valid_file_names + self.full_file_names}'
                                )
        return split_type
