import numpy as np
from dataclasses import dataclass

@dataclass 
class Neighbors:
    left: int = -1
    right: int = -1
    top: int = -1
    bottom: int = -1

    def determin_neighbors(self,rank,size):
        self.top = rank + int(np.sqrt(size))
        self.bottom = rank - int(np.sqrt(size))
        self.right = rank + 1
        self.left = rank - 1