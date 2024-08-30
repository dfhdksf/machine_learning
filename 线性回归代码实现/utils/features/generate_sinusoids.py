import numpy as np

#对特征进行非线性变换（sin）使其有非线性表达，从而使得结果有非线性形态
def generate_sinusoids(dataset, sinusoid_degree):
    """
    sin(x).
    """

    num_examples = dataset.shape[0]
    sinusoids = np.empty((num_examples, 0))

    for degree in range(1, sinusoid_degree + 1):
        sinusoid_features = np.sin(degree * dataset)
        sinusoids = np.concatenate((sinusoids, sinusoid_features), axis=1)
        
    return sinusoids
