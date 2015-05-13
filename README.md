#The unreasonable effectiveness of SGD

This is a repo for me to play with the parameters of SGD, which is an unintuitively effective way (for me) to maximize a function. The code 
does not look pretty but it works. Obviously the conclusions below don't generalize easily.

logistic_regression.py finds optimal coefficients for a logistic regression by maximizing the likelihood of the data. 
X are the labels and x are the features. Xs are drawn from a Bernoulli distribution with parameter theta. 
Each theta depends on cross sectional features x via a sigmoid link function. 
The likelihood of the data is minimized via
1. gradient descent 
2. stochastic gradient descent (SGD)
3. parallel SGD (as in "Parallelized Stochastic Gradient Descent" by Marty Zinkevich, Markus Weimer and Lihong Li)
4. adagrad

SGD converges (ie 10% error) in 1 epoch (actually after 2K points), gradient descent takes 16 steps. The learning rate for SGD is 2 orders
of magnitude smaller. The parallel version was not faster, due to the fact that all nodes had the same bias in the starting location.
Adagrad got quickly within 30%, then moved around. Maybe it would make sense to start with SGD and once you get close, do batches.

