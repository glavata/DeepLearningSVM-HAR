import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from time import perf_counter 
from statistics import mean
from pathlib import Path


class Pipeline:


    def __init__(self, X, y, z, dataset_name, cnn, svm):
        self.__X = X
        self.__y = y
        self.__z = z
        self.__dataset_name = dataset_name
        self.__cnn = cnn
        self.__svm = svm

        if not os.path.exists('temp_data'):
            os.makedirs('temp_data')

        if not os.path.exists('temp_data/batches'):
            os.makedirs('temp_data/batches')


    def train(self, folds = 10, num_epochs = 1, batch_count = 1):
        
        self.__folds = folds
        times_split = []

        labels = list(range(0,self.__z))
        conf_mat = np.zeros((self.__z,self.__z))
        acc_s = 0

        skf = StratifiedKFold(n_splits=folds, shuffle=True)
        i = 0

        self.__clear_folder('temp_data/')
        self.__clear_folder('temp_data/batches/')

        for train_split, test_split in skf.split(self.__X[0], self.__y):
            self.__save_train_test_split(train_split, test_split)
        
            self.__train_cnn(num_epochs, i, batch_count)
            self.__train_svm(i)

            post_X_test, post_y_test = self.__load_split(train = False)

            post_X_test = self.__cnn.get_out_data(post_X_test)
            y_pred = self.__svm.predict(post_X_test)

            acc_s += accuracy_score(post_y_test,y_pred)
            res_c = confusion_matrix(post_y_test, y_pred, labels=labels)
            conf_mat += res_c

            i+=1
            self.__cnn.reinit()
            self.__svm.reinit()

    
        acc_s /= folds
        
        #time_mins_avg_train_elapsed = mean(times_split)

        return [conf_mat, acc_s]


    def __train_cnn(self, num_epochs, tr_split_ind, batch_count):
        
        for e in range(0,num_epochs):
            X_train, y_train = self.__load_split(train = True)
            X_train_mat, X_train_vec = X_train[0], X_train[1]

            rand_ind = np.array(list(range(0,X_train_mat.shape[0])))
            np.random.shuffle(rand_ind)

            X_train_mat = X_train_mat[rand_ind]
            X_train_vec = X_train_vec[rand_ind]
            y_train = y_train[rand_ind]

            if(batch_count > 1):
                skf = KFold(n_splits=batch_count, shuffle=False)
                s = 0
                #save train x/y split in parts(batches)
                for _, part_ind in skf.split(X_train_mat, y_train):
                    X_part_mat = X_train_mat[part_ind]
                    X_part_vec = X_train_vec[part_ind]
                    y_part = y_train[part_ind]
                    self.__save_data_generic('temp_data/batches/' + self.__dataset_name + '_' + str(s), [X_part_mat, X_part_vec], y_part)
                    s+=1
            else:
                self.__save_data_generic('temp_data/batches/' + self.__dataset_name + '_0' , [X_train_mat, X_train_vec], y_train)

            for j in range(0, batch_count):
                #open train x/y split part(batch) and train
                X_batch, y_batch = self.__load_data_generic('temp_data/batches/' + self.__dataset_name + '_' + str(j))
                text = "TRAINING CNN: "
                res = self.__cnn.train_model(X_batch, y_batch)

                print("{0} val_split {1}/{2} epoch {3}/{4} batch {5}/{6}  ".format(text, tr_split_ind+1, self.__folds, \
                                                                                e+1, num_epochs, j+1,batch_count), \
                                                                                     end="\r", flush=True)

    def __train_svm(self, tr_split_ind):
        X_train, y_train = self.__load_split(train = True)
        text = "TRAINING SVM:"
        X_train = self.__cnn.get_out_data(X_train)
        self.__svm.train_model(X_train, y_train)
            
        
    def __clear_folder(self, folder):
        for parent, dirnames, filenames in os.walk(folder):
            for fn in filenames:
                os.remove(os.path.join(parent, fn))

    def __save_train_test_split(self, train_split, test_split):
        #save train x/y split
        with open('temp_data/' + self.__dataset_name + '_train_split', 'wb') as f:
            X_mat, X_vec = self.__X[0][train_split], self.__X[1][train_split]
            X_train, y_train = [X_mat, X_vec], self.__y[train_split]
            np.save(f, np.array([X_train, y_train]))

        #save test x/y split
        with open('temp_data/' + self.__dataset_name + '_test_split', 'wb') as f:
            X_mat, X_vec = self.__X[0][test_split], self.__X[1][test_split]
            X_test, y_test = [X_mat, X_vec], self.__y[test_split]
            np.save(f, np.array([X_test, y_test]))

    def __load_split(self, train):
        subfol = '_train_split' if train else '_test_split'
        X, y = self.__load_data_generic('temp_data/' + self.__dataset_name + subfol)

        return X, y

    def __load_data_generic(self, folder):
        res1, res2 = None, None
        with open(folder, 'rb') as f:
            rdy_arr = np.load(f, allow_pickle=True)
            res1, res2 = rdy_arr[0], rdy_arr[1]

        return res1, res2

    def __save_data_generic(self, dir_tgt, part1, part2):
        with open(dir_tgt, 'wb') as f:
            np.save(f, np.array([part1, part2]))