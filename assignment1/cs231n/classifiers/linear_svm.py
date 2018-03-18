import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  # gradient flowing from margins max(0,...) to scores
  dmargin_scores = np.zeros((num_train, num_classes))
  scores = np.zeros((num_train, num_classes))
  margin_all = np.zeros((num_train, num_classes))
  for i in xrange(num_train):
    scores[i] = X[i].dot(W)
    correct_class_score = scores[i,y[i]]
    for j in xrange(num_classes):
      margin = scores[i,j] - correct_class_score + 1 # note delta = 1
      if j == y[i]:
        continue
      if margin > 0:
        dmargin_scores[i,j] = 1
        dmargin_scores[i,y[i]] -= 1
        margin_all[i,j] = margin
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  #DxN, NxC
  dW = X.T.dot(dmargin_scores)/num_train
  dW += reg*2*W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  N,D = X.shape
  C = W.shape[1]

  scores = X.dot(W)
  row_idx = np.arange(0,N,1)
  col_idx = y.reshape((1,N))
  true_scores = scores[row_idx, col_idx].T
  delta = 1
  margin = np.maximum(0,scores-true_scores+delta)
  #no margin if weight belongs to true class
  margin[row_idx, col_idx]=0 
  penalty = reg * np.sum(W*W)
  loss = np.mean(np.sum(margin, axis=1)) + penalty
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  dmargin_score = np.zeros((N,C))
  dmargin_score[margin>0] = 1
  # in every margin row, check how many are positive
  dmargin_score[row_idx, col_idx] = -np.sum(margin>0,axis=1)
  dW = 1/N * X.T.dot(dmargin_score) 
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
