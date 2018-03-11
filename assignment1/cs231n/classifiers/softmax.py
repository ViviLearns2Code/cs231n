import numpy as np
from random import shuffle
from past.builtins import xrange

class Softmax:
    def __init__(self, num_features, num_classes):
        self.C = num_classes
        self.D = num_features
        # results of forward pass
        self.scores = np.zeros((1,num_classes))
        self.shift = 0
        self.stable = np.zeros((1,num_classes))
        self.denominator = 0
        self.nominator = 0
        self.result = 0
        # Pair (x,class_x)
        self.x = np.zeros((1, num_features))
        self.truth = 0
        # Debug mode
        self.debug_on = False

    def forward(self, x, class_x, W):
        ''' 
        Calculates the softmax expression for a given sample (x,class_x) 
        and weights W and additionally returns the back propagated weight
        Inputs:
        - W: A numpy array of shape (D, C) containing weights.
        - x: A numpy array of shape (1, D)
        - class_x: label of x (scalar)
        '''
        self.x = x
        self.truth = class_x
        self.log("Number of features:",self.D)
        self.log("Number of classes:",self.C)

        self.scores = np.dot(x,W) #1xC
        self.shift = (-1)*np.max(self.scores)
        self.stable = self.scores + self.shift
        self.denominator = np.sum(np.exp(self.stable))

        true_score = self.scores[:,class_x]
        stable_true = self.stable[:,class_x]
        self.nominator = np.exp(stable_true)

        self.result = self.nominator / self.denominator
        return self.result

    def backward(self, dProp):
        ''' calculate local gradients '''
        # gradient flowing through denominator
        ddenominator = dProp * self.nominator *(-1/self.denominator**2)
        self.log("Gradient through denominator gate", ddenominator)

        # gradient flowing through
        dnominator = dProp * (1/self.denominator)
        self.log("Gradient through nominator gate", dnominator)

        # gradient flowing through stable (now vector)
        # a) flow from denominator
        dstable = ddenominator * np.exp(self.stable)
        # b) flow from nominator
        dstable[:,self.truth] += dnominator * np.exp(self.stable[:,self.truth])
        self.log("Gradient through stable gate \n", dstable)

        # gradient flowing through shift (routed to 1 component)
        dshift = np.zeros((1,self.C))
        dshift[:,np.argmax(self.scores)] = (-1)*np.sum(dstable)
        self.log("Gradient through shift gate \n", dshift)

        # gradient flowing through scores
        # a) flow from shift
        dscores = dshift
        # b) flow from stable
        dscores += dstable
        self.log("Gradient through scores gate \n", dscores)

        # element-wise multiplication
        # DxC 1xC DxC 
        dW = np.tile(self.x.T,(1,self.C))*dscores
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
    N = np.size(X, axis=0)
    D = np.size(X, axis=1)
    C = np.size(W, axis=1)

    dcross_entropy = np.zeros_like(W)
    cross_entropy = 0
    for i in list(range(N)):
        softmax = Softmax(D,C)
        softmax.set_debug_mode(debug_on)
        softmax_result = softmax.forward(x=X[i,:][np.newaxis,:],class_x=y[i],W=W)
        cross_entropy -= 1/N * np.log(softmax_result)
        dcross_entropy += softmax.backward(dProp=1/N *(-1)*1/(softmax_result))
        if debug_on:
            print("Cross-Entropy derivative:\n", dcross_entropy)
    penalty = reg * np.sum(np.sum(np.square(W)))
    dpenalty = reg * 2 * W
    if debug_on:
        print("Penalty derivative:\n", dpenalty)
    loss = cross_entropy + penalty
    dW = dcross_entropy + dpenalty
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
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
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
