import os
import numpy as np
from cv2 import cv2
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pipeline import Pipeline
from models.cnn import CNN
from models.svm import SVM_C

from data_load import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

dataloader = DataLoader(Dataset.PKU_MMD,'PKUMMD', settings={'full':True})
X, y, z, v = dataloader.load_data( reload=False)

conf_mat = np.zeros((z,z))

cnn = CNN([], z,X[0].shape, X[1].shape)
svm = SVM_C([])

pipeline = Pipeline(X, y, z, "PKUMMD_full", cnn, svm)
results = pipeline.train(10, 10, 100)

print("Final accuracy: " + str(results[1]))
sn.heatmap(results[0], annot=True, annot_kws={"size": 10}, fmt='g') # font size
plt.show()
