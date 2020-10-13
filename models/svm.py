from sklearn import svm
from sklearn.base import clone

class SVM_C():
    
    def __init__(self, params):
        super().__init__()
        self.__params = params
        self.__define_model()

    def __define_model(self):
        params = self.__params
        #clf = svm.SVC(kernel=params['Kernel'], C=params['C'], degree=params['Degree'], gamma=params['Gamma'])
        clf = svm.SVC(kernel='poly', C=0.01, degree=1.5, gamma=6.8)
        self.__model = clf
        
    def train_model(self, X, y):
        self.__model.fit(X, y)

    def predict(self, y):
        return self.__model.predict(y)

    def reinit(self):
        self._model = clone(self.__model)
