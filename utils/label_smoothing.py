# import packages
import numpy as np


"""
- Label smoothing function
- Params:
    - labels, one-hot encoded labels
    - factor, smoothing factor, default set is 10%
"""
def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels
