""" The unreasonable effectiveness of SGD

Finds optimal coefficients for a logistic regression where X are the labels 
and x are the features. Xs are drawn from a Bernoulli distribution with parameter theta. 
Each theta depends on cross sectional features x via a sigmoid link function.

To find the coefficients it maximizes the likelihood of the data via gradient descent
and stochastic gradient descent.

SGD converges in 4 epochs, gradient descent takes 31 steps. The learning rate for SGD is 2 orders
of magnitude smaller. Obviously these conclusions don't generalize necessarily
"""

import math
import numpy as np

def logis(x):
    return 1.0/(1.0+math.exp(-x))

def bernoulli_draw(theta):
    if np.random.rand() < theta:
        return 1
    else:
        return 0

def loglik(X, x, alpha1, alpha2):
    alpha = np.transpose(np.matrix([alpha1, alpha2]))
    thetas = [logis(num) for num in x* alpha]
    log_prob = list()
    for theta, obs in zip(thetas, X): 
        if obs == 1:
            log_prob.append(math.log(theta))
        else: 
            log_prob.append(math.log(1.0-theta))
    return(sum(log_prob))

def gradient_log_lik(X, x, alpha1, alpha2):
    alpha = np.transpose(np.matrix([alpha1, alpha2]))
    thetas = [logis(num) for num in x* alpha]
    return((X - np.matrix(thetas)) * x)    


# generate the data
np.random.seed(2349)
alpha1 = 0.3
alpha2 = 0.9
alpha = np.transpose(np.matrix([alpha1, alpha2]))
N = 10000                                               # number of observations
range_x = 5                                             # range of values that featues can take
x = (np.random.rand(N,2) - 0.5) * range_x               # features
thetas = [logis(num) for num in x* alpha]               # Bernoulli theta for each observation
X = [ bernoulli_draw(theta) for theta in thetas]        # labels


# gradient descent
# convergence within 10% error after 16 iterations
learning_rate = 0.0001
start_alpha1hat = 5
start_alpha2hat = -5
n_steps = 20

alpha1hat = start_alpha1hat
alpha2hat = start_alpha2hat
for idx in range(n_steps):
    grad = gradient_log_lik(X, x, alpha1hat, alpha2hat)
    alpha1hat = alpha1hat + learning_rate * float(grad[0,0])
    alpha2hat = alpha2hat + learning_rate * float(grad[0,1])
    try:
        print idx, alpha1hat, alpha2hat, loglik(X, x, alpha1hat, alpha2hat)
    except ValueError:
        print idx, alpha1hat, alpha2hat

# stochastic gradient descent
# the learning rate must be set 2 orders of magnitude smaller otherwise algo jumps around
# convergence within 10% error after 1 epochs! Actually just needs 2500 observations
learning_rate = 0.000001
start_alpha1hat = 5
start_alpha2hat = -5
n_epochs = 2

alpha1hat = start_alpha1hat
alpha2hat = start_alpha2hat
for idx2 in range(n_epochs):
    for idx in range(len(X)):
        # multiply the gradient of one observation by the number of observations
        # so that the sgd gradient is analogous to the normal gradient that is the sum of 
        # the gradients of all the points
        grad = len(X)*gradient_log_lik(X[idx], x[idx], alpha1hat, alpha2hat)
        alpha1hat = alpha1hat + learning_rate * float(grad[0,0])
        alpha2hat = alpha2hat + learning_rate * float(grad[0,1])
        if idx % 100 == 0:
            try:
                print idx2, idx, alpha1hat, alpha2hat, grad[0,0], grad[0,1], loglik(X, x, alpha1hat, alpha2hat)
            except ValueError:
                print idx2, idx, alpha1hat, alpha2hat, grad[0,0], grad[0,1]

# parallel stochastic gradient descent
# each node sees part of the data
# this takes 2 epochs to converge, ie every node needs to see 2K observations, exactly
# as many as in the SGD case above. Averaging adds nothing.
# This makes sense since averaging happens while the algorithm is far from the minimum.
# Another way to say this is that the alphahat_ave has bias due to the same initialization
learning_rate = 0.000001
start_alpha1hat = 5
start_alpha2hat = -5
n_epochs = 100
n_machines=10
n_obs_machine = int(len(X)/n_machines)

alpha1hat = [start_alpha1hat for idx in range(n_machines)]
alpha2hat = [start_alpha2hat for idx in range(n_machines)]
for idx2 in range(n_epochs):
    for idx3 in range(n_machines):
        for idx in range( idx3 * n_obs_machine, (idx3+1) * n_obs_machine):        
            grad = len(X)*gradient_log_lik(X[idx], x[idx], alpha1hat[idx3], alpha2hat[idx3])
            alpha1hat[idx3] = alpha1hat[idx3] + learning_rate * float(grad[0,0])
            alpha2hat[idx3] = alpha2hat[idx3] + learning_rate * float(grad[0,1])

    alpha1hat_ave = sum(alpha1hat)/n_machines
    alpha2hat_ave = sum(alpha2hat)/n_machines
    try:
        print idx2, idx, alpha1hat_ave, alpha2hat_ave, grad[0,0], grad[0,1], loglik(X, x, alpha1hat_ave, alpha2hat_ave)
    except ValueError:
        print idx2, idx, alpha1hat_ave, alpha2hat_ave, grad[0,0], grad[0,1]


# stochastic gradient descent - adagrad
# Within the first 50 iterations it gets close, then keeps bouncing around
learning_rate = 1.0
tau0 = 1000
start_alpha1hat = 5
start_alpha2hat = -5
n_epochs = 10

alpha1hat = start_alpha1hat
alpha2hat = start_alpha2hat
s1 = 0
s2 = 0
for idx2 in range(n_epochs):
    for idx in range(len(X)):
        # multiply the gradient of one observation by the number of observations
        # so that the sgd gradient is analogous to the normal gradient that is the sum of 
        # the gradients of all the points
        grad = len(X)*gradient_log_lik(X[idx], x[idx], alpha1hat, alpha2hat)
        s1 = s1 + grad[0,0]**2
        s2 = s2 + grad[0,1]**2
        alpha1hat = alpha1hat + learning_rate * float(grad[0,0])/(tau0 + math.sqrt(s1))
        alpha2hat = alpha2hat + learning_rate * float(grad[0,1])/(tau0 + math.sqrt(s2))
        if idx % 100 == 0:
            try:
                print idx2, idx, alpha1hat, alpha2hat, grad[0,0], grad[0,1], tau0 + math.sqrt(s1), loglik(X, x, alpha1hat, alpha2hat)
            except ValueError:
                print idx2, idx, alpha1hat, alpha2hat, grad[0,0], grad[0,1]

# change into batch after a few steps, depending on










