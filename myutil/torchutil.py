import os
import torch
import numpy as np 
import random

class Reproducibility:
    """Ensure Reproducibility
    """
    def __init__(self, seed:int=42):
        self.seed = seed

    def seed_basic(self):
        """Ensure basic reproducibility

        Keyword Arguments:
            seed {int} -- (default: {42})
        """

        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)


    def seed_everything(self):
        """Ensure torch reproducibility

        Keyword Arguments:
            seed {int} -- (default: {42})
        """
        self.seed_basic()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(self.seed)