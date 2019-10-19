#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time         ：  2019/8/11 上午8:21
# @Author       ：  ModyfiAI
# @Email        ：  rongshunlin@126.com
# @File         ：  data.py
# @description  ：  仅供学习, 请勿用于商业用途
import re
import numpy as np
import pandas as pd

class DataSet(object):
    def __init__(self, positive_data_file, negative_data_file):
        self.x_text, self.y = self.load_data_and_labels(positive_data_file, negative_data_file)

    def load_data_and_labels(self, positive_data_file, negative_data_file):
        # load data from files
        if positive_data_file.endswith('.xls'):
            neg = pd.read_excel(negative_data_file, header=None)
            pos = pd.read_excel(positive_data_file, header=None)
            negative_data = [t[0].strip() for t in neg.values]
            positive_data = [t[0].strip() for t in pos.values]
        else:
            positive_data = list(open(positive_data_file, "r", encoding='utf-8').readlines())
            positive_data = [s.strip() for s in positive_data]
            negative_data = list(open(negative_data_file, "r", encoding='utf-8').readlines())
            negative_data = [s.strip() for s in negative_data]

        # split by words
        x_text = positive_data + negative_data

        x_text = [' '.join(list(sent)) for sent in x_text]

        # generate labels
        positive_labels = [[0, 1] for _ in positive_data]
        negative_labels = [[1, 0] for _ in negative_data]
        y = np.concatenate([positive_labels, negative_labels], 0)

        return [x_text, y]

    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                # print (shuffled_data[start_index])
                yield shuffled_data[start_index:end_index]
