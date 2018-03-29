import numpy as np
from random import shuffle
from past.builtins import xrange


class SoftmaxFun:
    def __init__(self, debug_on=False):
        '''
        Initialize Softmax Function Building Block
        '''
        self.debug_on = debug_on

    def forward(self,X,W,y=None):
        '''
        Calculates the softmax expression for each sample in (X,y) and weight matrix W
        Inputs:
        - X: NxD matrix with N samples and D features
        - y: Nx1 vector with true categories of each sample in X, encoded with {0,1...}
        - W: Weight matrix of dimension DxC
        Returns:
        - Softmax: If y is provded, Nx1 vector with evaluated softmax results per sample. 
        If y is not provided, NxC matrix with softmax results for every class and sample.
        '''
        self.C = np.size(W, axis=1)  # number of classes
        self.D = np.size(W, axis=0)
        self.N = np.size(X, axis=0)

        # results of forward pass
        self.scores = np.zeros((self.N, self.C))
        self.shift = np.zeros((self.N, 1))
        self.stable = np.zeros((self.N, self.C))
        self.stable_true = np.zeros((self.N, 1))
        self.denominator = np.zeros((self.N, 1))
        self.nominator = np.zeros((self.N, 1))
        self.result = np.zeros((self.N, 1))

        # Pair (X,y)
        self.X = X
        self.y = y
        self.W = W

        self.scores = self.X.dot(self.W)  # NxC
        self.shift = (-1)*np.max(self.scores, axis=1, keepdims=True)  # Nx1
        self.stable = self.scores + self.shift  # NxC
        self.denominator = np.sum(
            np.exp(self.stable), axis=1).reshape((self.N,1))  # Nx1
        
        if self.y is not None:
            # pick the score with col index corresponding y of the sample
            self.stable_true = self.stable[list(range(self.N)), self.y.T].reshape((self.N, 1))  # Nx1
            self.nominator = np.exp(self.stable_true)
            self.result = self.nominator / self.denominator
        else:
            self.result = self.stable / self.denominator #NxC        
        return self.result

    def backward(self, dProp):
        '''
        Calculate local gradients from forward pass and chain local gradient to incoming gradient
        Input:
        - dProp: Nx1 vector with incoming gradient per sample
        Returns:
        - dW: DxC matrix with backpropagated gradients w.r.t. weights
        '''
        # gradient flowing through denominator
        ddenominator = dProp * self.nominator * (-1/self.denominator**2)  # Nx1
        self.log("Gradient through denominator gate", ddenominator)

        # gradient flowing through
        dnominator = dProp * (1/self.denominator)  # Nx1
        self.log("Gradient through nominator gate", dnominator)

        # gradient flowing through stable (now vector)
        # a) flow from denominator
        dstable = ddenominator * np.exp(self.stable)  # NxC
        # b) flow from nominator
        row_indices = np.arange(0,self.N,1)
        col_indices = self.y.reshape((1,self.N))
        add = dnominator * np.exp(self.stable_true)
        dstable[row_indices, col_indices] += add.T[0, :]
        self.log("Gradient through stable gate \n", dstable)

        # gradient flowing through shift (routed to 1 component)
        dshift = np.zeros((self.N, self.C))  # NxC
        col_indices = [i for i in np.argmax(self.scores, axis=1).tolist()]
        dshift[row_indices, col_indices] = (-1)*np.sum(dstable, axis=1)
        self.log("Gradient through shift gate \n", dshift)

        # gradient flowing through scores
        # a) flow from shift
        dscores = dshift
        # b) flow from stable
        dscores += dstable
        self.log("Gradient through scores gate \n", dscores)

        dW = self.X.T.dot(dscores)  # DxN NxC
        return dW

    def set_debug_mode(self, debug_on=True):
        ''' Set debug mode to print debug messages '''
        self.debug_on = debug_on

    def log(self, *arg):
        ''' Print text in case debug mode is activated '''
        if self.debug_on:
            print(*arg)


def softmax_loss_naive(W, X, y, reg, debug_on=False):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N,D = X.shape
    dcross_entropy = np.zeros_like(W)
    cross_entropy = 0
    for i in list(range(N)):
        softmax = SoftmaxFun(debug_on=debug_on)
        softmax_result = softmax.forward(X=X[i, :].reshape((1,D)), y=y[i].reshape((1, 1)), W=W)
        cross_entropy -= 1/N * np.log(softmax_result)
        dcross_entropy += softmax.backward(dProp=1/N * (-1)*1/(softmax_result))
    penalty = reg * np.sum(W**2)
    dpenalty = reg * 2 * W

    loss = cross_entropy + penalty
    dW = dcross_entropy + dpenalty
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, debug_on=False):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    N = X.shape[0]
    softmax = SoftmaxFun(debug_on=debug_on)
    softmax_eval = softmax.forward(X=X, y=y.reshape((N,1)), W=W)  # Nx1
    cross_entropy = np.mean(-np.log(softmax_eval))
    penalty = reg * np.sum(W**2)
    loss = cross_entropy + penalty
    
    dcross_entropy = (-1)*(1/N)*(1/softmax_eval)  # Nx1
    dsoftmax = softmax.backward(dcross_entropy)  # DxC
    dpenalty = reg * 2 * W  # DxC

    dW = dsoftmax + dpenalty
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW