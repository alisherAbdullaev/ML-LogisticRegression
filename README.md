# This project provides an implementation of logistic regression algorithm. It supports an arbitary number of feature inputs and accepts un-encoded class labels.

***

### LogReg.py

1) The constructor ` __init__() ` takes parameter **self** that will be an instatnce of the LogReg class. The **X** and **y** parameters will store the training data and function, and `NLL()` calculates and returns the negative log-likelihood score for a proposed model. 

2) Method `predict_proba()` calculates and returns 1D array with the predicted probabilities on a scale from 0 to 1 and where the number of entries equal to the number of rows in **X**.

3) Method `predict()` returns the predicted class with the threshold being equal to 0.5.

4) Method `score()` calculates and returns the training accuracy.

5) Method `summary()` prints out the message with number of observations, best coeffecient estimates, Log-Likelihood, and accuracy.

6) Method `precision_recall()` calculates and prints out the calculated precisions and recall to tell us more about the model. 

***

### testing.ipynb




