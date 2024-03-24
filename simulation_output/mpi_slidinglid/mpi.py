import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import time
import sys
#import matplotlib.pyplot as plt

startime = time.time()

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

@dataclass
class SlidingLidMPI(Base):
    pos_x : int = -1
    pos_y : int = -1
    rank : int = -1
    size : int = -1
    base_grid:int  =-1
    mpi: MPI = None

    def get_postions_out_of_rank_size_quadratic(self,rank,size):
        return (rank % int(np.sqrt(size))),(rank // int(np.sqrt(size))) 

    def fill_mpi_struct_fields(self):
        self.pos_x,self.pos_y = self.get_postions_out_of_rank_size_quadratic(self.rank,self.size)
        self.boundary_info.set_boundary_info(self.pos_x,self.pos_y,int(np.sqrt(self.size))-1,int(np.sqrt(self.size))-1)
        self.neighbours_info.determin_neighbors(self.rank,self.size)
        self.d = np.ones((self.size_x, self.size_y))
        self.u = np.zeros((2,self.size_x, self.size_y))
        self.f = self.calcequi()
 
    def sliding_lid_mpi(self):
        for i in range(self.steps):
            self.stream()
            self.bounce_back_choosen()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            self.comunicate()
        full_grid = self.collapse_data()
        self.f=full_grid
        self.plotter(full_grid)
           
    def collapse_data(self):
        full_grid = np.ones((9, self.base_grid, self.base_grid))
        if self.rank == 0:
            full_grid[:,0:self.size_x-2,0:self.size_y-2] = self.f[:,1:-1,1:-1]
            temp = np.zeros((9,self.size_x-2,self.size_y-2))
            for i in range(1,self.size):
                self.mpi.Recv(temp,source = i,tag = i)
                x,y = self.get_postions_out_of_rank_size_quadratic(i,self.size)
                full_grid[:,(0 + (self.size_x-2)*x):((self.size_x-2) + (self.size_x-2)*x),(0 + (self.size_y-2)*y):((self.size_y-2) + (self.size_y-2)*y)] = temp
        else:
            self.mpi.Send(self.f[:,1:-1,1:-1].copy(),dest=0, tag = self.rank)
        return full_grid

    def comunicate(self):
        if not self.boundary_info.right:
            recvbuf = self.f[:, -1, :].copy()
            self.mpi.Sendrecv(self.f[:,-2, :].copy(), self.neighbours_info.right, recvbuf=recvbuf, sendtag = 10, recvtag = 20)
            self.f[:, -1, :] = recvbuf
        if not self.boundary_info.left:
            recvbuf = self.f[:, 0, :].copy()
            self.mpi.Sendrecv(self.f[:, 1, :].copy(), self.neighbours_info.left, recvbuf=recvbuf, sendtag = 20, recvtag = 10)
            self.f[:, 0, :] = recvbuf
        if not self.boundary_info.bottom:
            recvbuf = self.f[:,: ,0 ].copy()
            self.mpi.Sendrecv(self.f[:, :,1 ].copy(), self.neighbours_info.bottom, recvbuf=recvbuf, sendtag = 30, recvtag = 40)
            self.f[:, :, 0] = recvbuf
        if not self.boundary_info.top:
            recvbuf = self.f[:, :, -1].copy()
            self.mpi.Sendrecv(self.f[:, :, -2].copy(), self.neighbours_info.top, recvbuf=recvbuf, sendtag = 40, recvtag = 30)
            self.f[:, :, -1] = recvbuf
           
    def plotter(self,full_grid):
        if self.rank==0:
            #print("Making Image")
            self.f=full_grid
            savestring = "data/slidingLidmpi_fullgrid_nodes_"+str(self.size)+"_grid_"+str(self.base_grid)+"_steps_"+str(self.steps)+".npy"
            np.save(savestring,self.f)
            #self.calcdensity()
            #self.calcvelocity()
            #x = np.arange(0, self.base_grid)
            #y = np.arange(0, self.base_grid)
            #print(self.base_grid)
            #print(self.u.shape)
            #X, Y = np.meshgrid(x, y)
            #speed = np.sqrt(self.u[0].T ** 2 + self.u[1].T ** 2)
            #plt.streamplot(X, Y, self.u[0].T, self.u[1].T, color=speed, cmap=plt.cm.jet)
            #ax = plt.gca()
            #ax.set_xlim([0, self.base_grid + 1])
            #ax.set_ylim([0, self.base_grid + 1])
            #plt.title("Sliding Lid")
            #plt.xlabel("x-Position")
            #plt.ylabel("y-Position")
            #fig = plt.colorbar()
            #fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
            #savestring = "slidingLidmpi"+str(self.size)+".png"
            #plt.savefig(savestring)
            #plt.show()
            savestring = "data/slidingLidmpi_time_nodes_"+str(self.size)+"_grid_"+str(self.base_grid)+"_steps_"+str(self.steps)+".txt"
            f = open(savestring,"w")
            totaltime = time.time() - startime
            f.write(str(totaltime))
            f.close()

def call():
        steps = int(sys.argv[2])
        re = 1000
        base_length = int(sys.argv[1])
        uw = 0.1
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank_1d = int(np.sqrt(size)) 
        if rank_1d*rank_1d != size:
            return RuntimeError
        slmpi=SlidingLidMPI(rank=comm.Get_rank(),size=comm.Get_size(),size_x=base_length//(rank_1d) + 2,size_y=base_length//(rank_1d) + 2,relaxation=((2 * re) / (6 * base_length * uw + re)),steps=steps,velocity_lid=uw,base_grid=base_length,mpi=comm)
        slmpi.fill_mpi_struct_fields()
        slmpi.sliding_lid_mpi()

call()
