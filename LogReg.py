import numpy as np 
from scipy.optimize import minimize 

class LogReg:
    def __init__(self, X, y):
        
        #Class attributes
        self.X = np.array(X)
        self.y = np.array(y)
        
        #Number of training samples
        self.N = self.X.shape[0]
        
        #List of mames of the two classes
        self.classes = np.unique(self.y)
        
        def NLL(beta):
            
            ones = np.ones((self.N, 1)) 
            X_ = np.hstack((ones, self.X))
            z = np.dot(X_, beta)
            z = z.reshape(-1,)
            p = (1 / (1 + np.exp(z))) 
            pi = np.where(self.y == self.classes[1], 1-p, p)
            nll = np.sum(-np.log(pi))
            return nll 
        
        #Best coefficients
        beta_guess = np.zeros(self.X.shape[1] + 1)
        min_results = minimize(NLL, beta_guess)
        self.coefficients = min_results.x
        
        #Training loss for the optimal model
        self.loss = NLL(self.coefficients)
        
        #Training accuracy
        self.accuracy = round(self.score(X, y),4)
        
    
    def predict_proba(self, X):
        
        self.X = np.array(X)
        ones = np.ones((self.X.shape[0], 1))
        X_ = np.hstack((ones, X))
        z = np.dot(X_, self.coefficients)
        z = z.reshape(-1,)
        p = (1 / (1 + np.exp(-z))) 
        return p
    
    
    def predict(self, X, t = 0.5):
        
        self.X = np.array(X)
        out = np.where(self.predict_proba(X) < t, self.classes[0], self.classes[1])
        return out
    
    
    def score(self, X, y, t = 0.5):
        
        self.X = np.array(X)
        self.y = np.array(y)
        accuracy = np.sum(self.predict(X,t) == self.y) / len(self.y)
        return accuracy
    
    
    def summary(self):
        
        print('+-------------------------------+')
        print('|  Logistic Regression Summary  |')
        print('+-------------------------------+')
        print('Number of training observations: ' + str(self.N))
        print('Coefficient Estimated: ' + str(self.coefficients))
        print('Negative Log-likelihood: ' + str(np.around(self.loss, decimals = 4)))
        print('Accuracy: ' + str(self.accuracy))
        
    
    
    def precision_recall(self, X, y, t = 0.5):
        
        self.X = np.array(X)
        self.y = np.array(y)
        
        #True positives for class 0
        X0 = np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[0]))
        
        #False positives for class 0
        X01 = np.sum((self.predict(X,t) == self.classes[0]) &
                             (self.y == self.classes[1]))
        
        #True positives for class 1
        X1 = np.sum((self.predict(X,t) == self.classes[1]) &
                            (self.y == self.classes[1]))
        
        #False positives for class 0
        X10 = np.sum((self.predict(X,t) == self.classes[1]) &
                             (self.y == self.classes[0]))
        
        precision0 =  round(X0 / (X0 + X01), 4)
        recall0 = round(X0 / np.sum(self.y == self.classes[0]), 4)
        
        precision1 =  round(X1 / (X1 + X10), 4)
        recall1 = round(X1 / np.sum(self.y == self.classes[1]), 4)
        
        print('Class: ' + str(self.classes[0]))
        print('  Precision = ' + str(precision0))
        print('  Recall    = ' + str(recall0))
        print('Class: ' + str(self.classes[1]))
        print('  Precision = ' + str(precision1))
        print('  Recall    = ' + str(recall1))
 

    def confusion_matrix(self, X, y, t = 0.5):
        
        conf = np.zeros(shape = (2,2), dtype = 'int')
        res = np.where(self.predict_proba(X) > t, self.classes[1], self.classes[0])
        
        for i in range(0, len(res)):
            if res[i] == self.classes[0]:
                if self.y[i] == self.classes[0]:
                    conf[0,0] += 1
                else:
                    conf[1,0] += 1
            else:
                 if self.y[i] == self.classes[0]:
                     conf[0,1] += 1
                 else:
                     conf[1,1] += 1            
        self.conf = conf
        return self.conf
