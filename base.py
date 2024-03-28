import numpy as np
from dataclasses import dataclass
from boundary import Boundary
from neighbor import Neighbors

@dataclass
class Base:
    # -------------------------- DATA MEMBERS --------------------------------
    # Lattice Dimensions
    size_x: int = -1
    size_y: int = -1
    # Probability Density Function = f(c,x,y) where x and y are dimensions of the lattice 
    # and c is the corresponding velocity vector for the D2Q9 model
    f: np = None
    # Density = d(x,y) where x and y are the dimensions of the lattice
    d: np = None
    # Velocity = u(2,x,y) where x and y are the dimensions of the lattice
    # where 2 denotes the x and y component of the velocity vector of the lattice grid
    u: np = None
    # Relaxation constant (omega)
    relaxation: float = 0.5
    # No of steps to be simulated for a fluid profile
    steps: int = 0
    # No of steps after which we get intermediate update
    every: int = 0
    # Velocity of Lid in case of moving wall
    velocity_lid: float = 0
    # Boundary Class object
    boundary_info: Boundary = Boundary()
    # Neighbours Class object
    neighbours_info: Neighbors = Neighbors()
    # Velocity Vector having 9 pairs of values according to the D2Q9 model
    velocity_vector: np.ndarray = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1], [0, 0, 1, 0, -1, 1, 1, -1, -1]]).T
    # Weights array for the collision term
    weights: np.ndarray = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T
    
    # ------------------------ MEMBER FUNCTIONS --------------------------------

    def calcdensity(self):
        """Calculate fluid density at each lattice point."""
        self.d = self.f.sum(axis=0)

    def calcvelocity(self):
        """Calculate fluid velocity at each lattice point."""
        self.u = (np.dot(self.f.T, self.velocity_vector).T / self.d)

    def calcequi(self):
        """Calculate equilibrium distribution function."""
        return (((1 + 3 * (np.dot(self.u.T, self.velocity_vector.T).T) + 9/2 * ((np.dot(self.u.T, self.velocity_vector.T) ** 2).T) - 3/2 * (np.sum(self.u ** 2, axis=0))) * self.d).T * self.weights).T

    def calccollision(self):
        """Perform collision step updating the distribution function."""
        self.f -= self.relaxation * (self.f - self.calcequi())

    def stream(self):
        """Stream particles along the velocity vectors."""
        for i in range(1, 9):
            self.f[i] = np.roll(self.f[i], self.velocity_vector[i], axis=(0, 1))

    def bounce_back_choosen(self):
        """Apply bounce-back boundary conditions."""
        if self.boundary_info.bottom:
            self.f[[2, 5, 6], :, 1] = self.f[[4, 7, 8], :, 0]
        if self.boundary_info.top:
            self.f[4, :, -2] = self.f[2, :, -1]
            self.f[7, :, -2] = self.f[5, :, -1] - 1 / 6 * self.velocity_lid
            self.f[8, :, -2] = self.f[6, :, -1] + 1 / 6 * self.velocity_lid
        if self.boundary_info.right:
            self.f[[3, 6, 7], -2, :] = self.f[[1, 8, 5], -1, :]
        if self.boundary_info.left:
            self.f[[1, 5, 8], 1, :] = self.f[[3, 7, 6], 0, :]
