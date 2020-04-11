## import packages
import numpy as np



def reweight_by_frequency(labels):
    """compute class weights w.r.t # of samples in each  class
    Args:
        - labels: one-hot encoded labels
    Returns: 
        - class weights
    """
    classTotals = labels.sum(axis=0)
    class_weights = classTotals.max() / classTotals
    return class_weights


def class_balance_by_effective_number(labels):
    """compute effective number of samples & corresponding weighting factors
    Args:
        - labels: one-hot encoded labels
    Returns: 
        - computed weighting factors of each class
    """
    classTotals = labels.sum(axis=0)  
    betas = (classTotals - 1) / classTotals 
    effective_nums = (1 - np.power(betas, classTotals)) / (1 - betas)
    class_weigts = 1 / effective_nums
    return class_weigts



