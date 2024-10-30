import numpy as np
class Model:
    def __init__(self):
        self.W = np.zeros((self.num_classes, self.input_size))
        self.b = np.zeros((self.num_classe,1))