import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time

def DataSize(data):
    negative_count = 0
    positive_count = 0
    for year in data:
        for a_company, value in year.items():
            if value[1][0] == 0:
                negative_count = negative_count + 1
            else:
                positive_count = positive_count + 1

    return negative_count, positive_count

def read_CSV(file_name, DEVICE, has_title=False):
    year = {}
    X = {}
    Y = {}
    file = open(file_name, 'r')
    if has_title:
        line = file.readline()
    count = 0
    line = file.readline()
    while line:
        #print("line ", str(count))
        temp = line.split(",")
        if not temp[0] in year:
            year[temp[0]] = []
            X[temp[0]] = []
            Y[temp[0]] = []
        year[temp[0]].append(int(temp[2]))
        aLine = []
        for i in range(18):
            aLine.append(float(temp[i + 3]))
        X[temp[0]].append(np.array(aLine,dtype=np.float32))
        if temp[1] == "alive":
            Y[temp[0]].append(0)
        else:
            Y[temp[0]].append(1)

        line = file.readline()
        count = count + 1
    return year,X,Y

def SplitTVT(year,X,Y, finalYear, middleYear, excluded):
    lastest_data = [{} for i in range(21)]
    middle_data = [{} for i in range(21)]
    early_data = [{} for i in range(21)]
    for key,value in year.items():
        l = len(value)
        if l > 3:
            if value[-1] not in excluded:
                if value[-1] > finalYear:
                    lastest_data[l][key] = (X[key],Y[key])
                elif value[-1] > middleYear:
                    middle_data[l][key] = (X[key],Y[key])
                else:
                    early_data[l][key] = (X[key],Y[key])
    return early_data,middle_data,lastest_data

class DataGenerator(IterableDataset):
    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        file_name,
        DEVICE,

        has_title=False,

    ):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.batch_type = batch_type

        #train_data, test_support_data, test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title),2008,2007,(2018,2017,2016,2015,2014,2013,2012,2011,2010))
        train_data, test_support_data,test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title),2007,2006,(2018,2017,2016,2015,2014,2013,2012,2011,2010,2009))
        #train_data, test_support_data, test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title), 2017, 2016,())

        negative_count, positive_count = DataSize(train_data)
        print("train_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))
        negative_count, positive_count = DataSize(test_support_data)
        print("test_support_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))
        negative_count, positive_count = DataSize(test_query_data)
        print("test_query_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))

        self.train_data = {}
        self.train_positive_data = {}
        self.train_negative_data = {}

        self.test_support_data = {}
        self.test_support_positive_data = {}
        self.test_support_negative_data = {}
        self.test_query_data = {}

        for year in train_data:
            for company, data in year.items():
                self.train_data[company] = data
                if data[1][0] == 0:
                    self.train_negative_data[company] = data
                else:
                    self.train_positive_data[company] = data

        for year in test_support_data:
            for company, data in year.items():
                self.test_support_data[company] = data
                if data[1][0] == 0:
                    self.test_support_negative_data[company] = data
                else:
                    self.test_support_positive_data[company] = data

        for year in test_query_data:
            for company, data in year.items():
                self.test_query_data[company] = data
    #MAML
    def _sample(self):
        #return self._Rsample()
        return self._UPsample()
    def _Rsample(self):
        if self.batch_type == "train":
            # shuffle the label when training
            if random.randint(0, 1) == 1:
                negative_label = np.array([1], dtype=np.float32)
                positive_label = np.array([0], dtype=np.float32)
            else:
                negative_label = np.array([0], dtype=np.float32)
                positive_label = np.array([1], dtype=np.float32)

            # the support have 50% to have each class
            support_label_batch = ()
            support_data_batch = ()
            if random.randint(0, 1) == 1:
                sampled_negative_data = random.sample(list(self.train_negative_data.values()),1)
                support_label_batch = support_label_batch + (np.copy(negative_label),)
                support_data_batch = support_data_batch + (sampled_negative_data[0][0][-1],)
            else:
                sampled_positive_data = random.sample(list(self.train_positive_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(positive_label),)
                support_data_batch = support_data_batch + (sampled_positive_data[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()
            sampled_data = random.sample(list(self.train_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if sampled_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (sampled_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))
        # test data
        else:
            negative_label = np.array([0], dtype=np.float32)
            positive_label = np.array([1], dtype=np.float32)

            #support
            support_label_batch = ()
            support_data_batch = ()

            if random.randint(0, 1) == 1:
                s = random.sample(list(self.test_support_negative_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(negative_label),)
                support_data_batch = support_data_batch + (s[0][0][-1],)
            else:
                s = random.sample(list(self.test_support_positive_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(positive_label),)
                support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            query_data = random.sample(list(self.test_query_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if query_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (query_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))

    #MAML
    def _UPsample(self):
        if self.batch_type == "train":
            # shuffle the label when training
            if random.randint(0, 1) == 1:
                negative_label = np.array([1], dtype=np.float32)
                positive_label = np.array([0], dtype=np.float32)
            else:
                negative_label = np.array([0], dtype=np.float32)
                positive_label = np.array([1], dtype=np.float32)

            # the support have 50% to have each class
            support_label_batch = ()
            support_data_batch = ()
            if random.randint(0, 1) == 1:
                sampled_negative_data = random.sample(list(self.train_negative_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(negative_label),)
                support_data_batch = support_data_batch + (sampled_negative_data[0][0][-1],)
            else:
                sampled_positive_data = random.sample(list(self.train_positive_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(positive_label),)
                support_data_batch = support_data_batch + (sampled_positive_data[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()
            for i in range(self.num_samples_per_class - 1):
                if random.randint(0, 1) == 1:
                    sampled_negative_data = random.sample(list(self.train_negative_data.values()),1)
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                    query_data_batch = query_data_batch + (sampled_negative_data[0][0][-1],)
                else:
                    sampled_positive_data = random.sample(list(self.train_positive_data.values()), 1)
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                    query_data_batch = query_data_batch + (sampled_positive_data[0][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))
        # test data
        else:
            negative_label = np.array([0], dtype=np.float32)
            positive_label = np.array([1], dtype=np.float32)

            #support
            support_label_batch = ()
            support_data_batch = ()

            if random.randint(0, 1) == 1:
                s = random.sample(list(self.test_support_negative_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(negative_label),)
                support_data_batch = support_data_batch + (s[0][0][-1],)
            else:
                s = random.sample(list(self.test_support_positive_data.values()), 1)
                support_label_batch = support_label_batch + (np.copy(positive_label),)
                support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            query_data = random.sample(list(self.test_query_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if query_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (query_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))


class ProtoDataGenerator(IterableDataset):

    def __init__(
        self,
        num_classes,
        num_samples_per_class,
        batch_type,
        file_name,
        DEVICE,

        has_title=False,

    ):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.batch_type = batch_type

        #train_data, test_support_data, test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title),2008,2007,(2018,2017,2016,2015,2014,2013,2012,2011,2010))
        train_data, test_support_data,test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title),2007,2006,(2018,2017,2016,2015,2014,2013,2012,2011,2010,2009))
        #train_data, test_support_data, test_query_data = SplitTVT(*read_CSV(file_name, DEVICE, has_title), 2017, 2016,())

        negative_count, positive_count = DataSize(train_data)
        print("train_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))
        negative_count, positive_count = DataSize(test_support_data)
        print("test_support_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))
        negative_count, positive_count = DataSize(test_query_data)
        print("test_query_data size: negative_count ", str(negative_count), " positive_count ", str(positive_count))

        self.train_data = {}
        self.train_positive_data = {}
        self.train_negative_data = {}

        self.test_support_data = {}
        self.test_support_positive_data = {}
        self.test_support_negative_data = {}
        self.test_query_data = {}

        for year in train_data:
            for company, data in year.items():
                self.train_data[company] = data
                if data[1][0] == 0:
                    self.train_negative_data[company] = data
                else:
                    self.train_positive_data[company] = data

        for year in test_support_data:
            for company, data in year.items():
                self.test_support_data[company] = data
                if data[1][0] == 0:
                    self.test_support_negative_data[company] = data
                else:
                    self.test_support_positive_data[company] = data

        for year in test_query_data:
            for company, data in year.items():
                self.test_query_data[company] = data

    def _sample(self):
        #return self._Rsample()
        return self._UPsample()
    # ProtoNet
    def _Rsample(self):
        if self.batch_type == "train":
            # shuffle the label when training
            if random.randint(0, 1) == 1:
                negative_label = np.array([1], dtype=np.int64)
                positive_label = np.array([0], dtype=np.int64)
            else:
                negative_label = np.array([0], dtype=np.int64)
                positive_label = np.array([1], dtype=np.int64)

            # support
            support_label_batch = ()
            support_data_batch = ()

            s = random.sample(list(self.train_negative_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(negative_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            s = random.sample(list(self.train_positive_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(positive_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            sampled_data = random.sample(list(self.train_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if sampled_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (sampled_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))
        # test data
        else:
            negative_label = np.array([0], dtype=np.int64)
            positive_label = np.array([1], dtype=np.int64)

            #support
            support_label_batch = ()
            support_data_batch = ()


            s = random.sample(list(self.test_support_negative_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(negative_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            s = random.sample(list(self.test_support_positive_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(positive_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            query_data = random.sample(list(self.test_query_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if query_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (query_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))

    #ProtoNet
    def _UPsample(self):
        if self.batch_type == "train":
            # shuffle the label when training
            if random.randint(0, 1) == 1:
                negative_label = np.array([1], dtype=np.int64)
                positive_label = np.array([0], dtype=np.int64)
            else:
                negative_label = np.array([0], dtype=np.int64)
                positive_label = np.array([1], dtype=np.int64)

            # support
            support_label_batch = ()
            support_data_batch = ()

            s = random.sample(list(self.train_negative_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(negative_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            s = random.sample(list(self.train_positive_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(positive_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            for i in range(self.num_samples_per_class - 1):
                if random.randint(0, 1) == 1:
                    sampled_negative_data = random.sample(list(self.train_negative_data.values()),1)
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                    query_data_batch = query_data_batch + (sampled_negative_data[0][0][-1],)
                else:
                    sampled_positive_data = random.sample(list(self.train_positive_data.values()), 1)
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                    query_data_batch = query_data_batch + (sampled_positive_data[0][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))
        # test data
        else:
            negative_label = np.array([0], dtype=np.int64)
            positive_label = np.array([1], dtype=np.int64)

            #support
            support_label_batch = ()
            support_data_batch = ()


            s = random.sample(list(self.test_support_negative_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(negative_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            s = random.sample(list(self.test_support_positive_data.values()), 1)
            support_label_batch = support_label_batch + (np.copy(positive_label),)
            support_data_batch = support_data_batch + (s[0][0][-1],)

            # query
            query_label_batch = ()
            query_data_batch = ()

            query_data = random.sample(list(self.test_query_data.values()), self.num_samples_per_class - 1)
            for i in range(self.num_samples_per_class - 1):
                if query_data[i][1][0] == 0:
                    query_label_batch = query_label_batch + (np.copy(negative_label),)
                else:
                    query_label_batch = query_label_batch + (np.copy(positive_label),)
                query_data_batch = query_data_batch + (query_data[i][0][-1],)

            return (torch.from_numpy(np.stack(support_data_batch)), torch.from_numpy(np.stack(support_label_batch)), torch.from_numpy(np.stack(query_data_batch)),torch.from_numpy(np.stack(query_label_batch)))
