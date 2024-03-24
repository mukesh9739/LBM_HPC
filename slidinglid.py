import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class slidinglid(Base):
    def plotter(self,i):
        plt.clf()
        x = np.arange(0, self.size_x-2)
        y = np.arange(0, self.size_y-2)
        X, Y = np.meshgrid(x, y)
        speed = np.sqrt(self.u[0,2:-2,2:-2].T ** 2 + self.u[1,2:-2,2:-2].T ** 2)
        plt.streamplot(X,Y,self.u[0,2:-2,2:-2].T,self.u[1,2:-2,2:-2].T, color=speed,  cmap= plt.cm.jet)
        ax = plt.gca()
        ax.set_xlim([0, self.size_x+1])
        ax.set_ylim([0, self.size_y+1])
        titleString = "Sliding Lid (Gridsize " + "{}".format(self.size_x) + "x" + "{}".format(self.size_y)
        titleString += ",  $\\omega$ = {:.2f}".format(self.relaxation) + ", steps = {}".format(i) + ")"
        plt.title(titleString)
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        fig = plt.colorbar()
        fig.set_label("Velocity u(x,y,t)", rotation=270,labelpad = 15)
        savestring="simulation_output/slidinglid/slidinglid_step_"+str(i)+".png"
        plt.savefig(savestring)
        

    def simulation(self):
        self.d=np.ones((self.size_x+2,self.size_y+2))
        self.u=np.zeros((2,self.size_x+2,self.size_y+2))
        self.f=self.calcequi()
        for i in range(self.steps):
            self.stream()
            self.bounce_back_choosen() 
            self.calcdensity()
            self.calcvelocity()
            self.calccollision() 
            if (i % self.every) == 0 and i > 0:
                self.every = self.every*2
                self.plotter(i)
        self.plotter(self.steps)
        plt.show()
        
re=1000
base_length=300
uw=0.1
sl=slidinglid(size_x=base_length,size_y=base_length,every=100,steps=100000,relaxation=((2 * re) / (6 * base_length * uw + re)),velocity_lid=uw,boundary_info=Boundary(True,True,True,True))
sl.simulation()