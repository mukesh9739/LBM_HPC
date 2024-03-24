import numpy as np
from dataclasses import dataclass

@dataclass
class Boundary:
    left : bool = False
    right: bool = False
    top: bool = False
    bottom: bool = False

    def set_boundary_info(self,pox,poy,max_x,max_y):
        self.left = pox == 0
        self.bottom = poy == 0
        self.right = pox == max_x
        self.top = poy == max_y