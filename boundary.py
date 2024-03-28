import numpy as np
from dataclasses import dataclass

@dataclass
class Boundary:
    # Boundary condition flags for the lattice grid
    # ---------------------------------------------
    # Indicates if the left boundary is active (True) or not (False)
    left: bool = False
    # Indicates if the right boundary is active (True) or not (False)
    right: bool = False
    # Indicates if the top boundary is active (True) or not (False)
    top: bool = False
    # Indicates if the bottom boundary is active (True) or not (False)
    bottom: bool = False

    def set_boundary_info(self, pox: int, poy: int, max_x: int, max_y: int):
        """
        Determines the active boundaries based on the position of the lattice grid.

        Parameters:
        - pox (int): Position index along the x-axis of the lattice grid.
        - poy (int): Position index along the y-axis of the lattice grid.
        - max_x (int): Maximum index along the x-axis of the lattice grid.
        - max_y (int): Maximum index along the y-axis of the lattice grid.

        Sets the boundary conditions (left, right, top, bottom) as True if the lattice grid 
        is positioned at the respective extremities.
        """
        self.left = pox == 0  # Left boundary active if position is at the beginning of the x-axis
        self.bottom = poy == 0  # Bottom boundary active if position is at the beginning of the y-axis
        self.right = pox == max_x  # Right boundary active if position is at the end of the x-axis
        self.top = poy == max_y  # Top boundary active if position is at the end of the y-axis
