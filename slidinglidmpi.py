import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import time
import sys
from base import Base
#import matplotlib.pyplot as plt

startime = time.time()

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
            print("Making Image")
            self.f=full_grid
            savestring = "simulation_output/mpi_slidinglid/slidingLidmpi_fullgrid_nodes_"+str(self.size)+"_grid_"+str(self.base_grid)+"_steps_"+str(self.steps)+".npy"
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
            savestring = "simulation_output/mpi_slidinglid/slidingLidmpi_time_nodes_"+str(self.size)+"_grid_"+str(self.base_grid)+"_steps_"+str(self.steps)+".txt"
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