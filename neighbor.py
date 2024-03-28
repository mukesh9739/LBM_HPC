import numpy as np
from dataclasses import dataclass

@dataclass
class Neighbors:
    # Neighbor indices for a given lattice node
    # -----------------------------------------
    # Index of the neighboring node to the left
    left: int = -1
    # Index of the neighboring node to the right
    right: int = -1
    # Index of the neighboring node to the top
    top: int = -1
    # Index of the neighboring node to the bottom
    bottom: int = -1

    def determin_neighbors(self, rank: int, size: int):
        """
        Determines the indices of neighboring nodes based on the current node's rank and 
        the total size (number of nodes) of the lattice grid.

        Parameters:
        - rank (int): The rank (index) of the current node in the lattice grid.
        - size (int): The total number of nodes in the lattice grid.

        The method calculates the neighbors' indices considering a square lattice layout, 
        where `size` is expected to be a perfect square to correctly compute top and bottom neighbors.
        """
        # Calculate the top neighbor's index by moving up one row in a square grid
        self.top = rank + int(np.sqrt(size))
        # Calculate the bottom neighbor's index by moving down one row in a square grid
        self.bottom = rank - int(np.sqrt(size))
        # Calculate the right neighbor's index; assumes periodic boundary conditions
        self.right = rank + 1
        # Calculate the left neighbor's index; assumes periodic boundary conditions
        self.left = rank - 1
