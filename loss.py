import numpy as np

class focal_binary_Loss:
    def __init__(self, gamma):
        '''
        :param gamma: The parameter to specify the gamma indicator
        '''
        self.gamma = gamma

    def robust_pow(self, num_base, num_pow):
        # numpy does not permit negative numbers to fractional power
        # use this to perform the power algorithmic

        return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

    def focal_binary_object(self, pred, dtrain):
        gamma = self.gamma
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        # complex gradient with different parts
        g1 = sigmoid_pred * (1 - sigmoid_pred)
        g2 = label + ((-1) ** label) * sigmoid_pred
        g3 = sigmoid_pred + label - 1
        g4 = 1 - label - ((-1) ** label) * sigmoid_pred
        g5 = label + ((-1) ** label) * sigmoid_pred
        # combine the gradient
        grad = gamma * g3 * self.robust_pow(g2, gamma) * np.log(g4 + 1e-9) + \
               ((-1) ** label) * self.robust_pow(g5, (gamma + 1))
        # combine the gradient parts to get hessian components
        hess_1 = self.robust_pow(g2, gamma) + \
                 gamma * ((-1) ** label) * g3 * self.robust_pow(g2, (gamma - 1))
        hess_2 = ((-1) ** label) * g3 * self.robust_pow(g2, gamma) / g4
        # get the final 2nd order derivative
        hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma +
                (gamma + 1) * self.robust_pow(g5, gamma)) * g1

        return grad, hess

    
class weight_binary_Cross_Entropy:
    def __init__(self, imbalance_alpha):
        '''
        :param imbalance_alpha: the imbalanced alpha value for the minority class (label as '1')
        '''
        self.imbalance_alpha = imbalance_alpha

    def weighted_binary_cross_entropy(self, pred, dtrain):
        # assign the value of imbalanced alpha
        imbalance_alpha = self.imbalance_alpha
        # retrieve data from dtrain matrix
        label = dtrain.get_label()
        # compute the prediction with sigmoid
        sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
        # gradient
        grad = -(imbalance_alpha ** label) * (label - sigmoid_pred)
        hess = (imbalance_alpha ** label) * sigmoid_pred * (1.0 - sigmoid_pred)

        return grad, hess
