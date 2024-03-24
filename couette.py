import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class couette(Base):
    def simulation(self):
        self.d=np.ones((self.size_x,self.size_y+2))
        self.u=np.zeros((2,self.size_x,self.size_y+2))
        self.f=self.calcequi()
        for i in range(self.steps):
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            self.stream()
            self.bounce_back_choosen()   
            if (i % self.every) == 0 and i > 0:
                self.every = self.every*2
                plt.plot(self.u[0, 50, 1:-1],np.arange(0,50),label = "Step {}".format(i))
        y = self.velocity_lid*1/50*np.arange(0,50)
        plt.plot(y,np.arange(0,50), label ="Analytical")
        xmin,xmax,ymin,ymax=plt.axis()
        plt.plot([xmin,xmax],[ymin+2,ymin+2], color='k')
        plt.plot([xmin,xmax],[ymax-2,ymax-2], color='r')
        plt.legend()
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Couette flow (Gridsize 100x52, $\omega = 0.5$)')
        plt.savefig("simulation_output/couette/CouetteFlow.png")
        plt.show()

c=couette(size_x=100,size_y=50,steps=5000,every=10,velocity_lid=0.1,boundary_info=Boundary(False,False,True,True))
c.simulation()