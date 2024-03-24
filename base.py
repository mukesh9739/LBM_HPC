import numpy as np
from dataclasses import dataclass
from boundary import Boundary
from neighbor import Neighbors

@dataclass
class Base:
    size_x: int=-1
    size_y: int=-1
    f: np=None
    d: np=None
    u: np=None
    relaxation: float=0.5
    steps: int=0
    every: int=0
    velocity_lid: float=0
    boundary_info: Boundary=Boundary()
    neighbours_info: Neighbors=Neighbors()
    velocity_vector: np.ndarray = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
    weights: np.ndarray = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T

    def calcdensity(self):
        self.d= self.f.sum(axis=0)

    def calcvelocity(self):
        self.u= (np.dot(self.f.T, self.velocity_vector).T / self.d) 
    
    def calcequi(self):
        return ( ( ( 1 + 3*(np.dot(self.u.T,self.velocity_vector.T).T) + 9/2*((np.dot(self.u.T,self.velocity_vector.T) ** 2).T) - 3/2*(np.sum(self.u ** 2, axis=0)) ) * self.d ).T * self.weights ).T

    def calccollision(self):  
        self.f -= self.relaxation * (self.f-self.calcequi())    

    def stream(self):
        for i in range(1,9):
            self.f[i] = np.roll(self.f[i],self.velocity_vector[i], axis = (0,1))
    
    def bounce_back_choosen(self):
        if self.boundary_info.right:
            self.f[[3,6,7],-2,:]=self.f[[1,8,5],-1,:]
        if self.boundary_info.left:
            self.f[[1,5,8],1,:]=self.f[[3,7,6],0,:]
        if self.boundary_info.bottom:
            self.f[[2,5,6],:,1]=self.f[[4,7,8],:,0]
        if self.boundary_info.top:
            self.f[4, :, -2] = self.f[2, :, -1]
            self.f[7, :, -2] = self.f[5, :, -1] - 1 / 6 * self.velocity_lid
            self.f[8, :, -2] = self.f[6, :, -1] + 1 / 6 * self.velocity_lid