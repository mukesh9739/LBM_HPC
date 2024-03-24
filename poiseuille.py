import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class poiseuille(Base):
    rho_null = 1
    diff=0.001
    rho_in = rho_null+diff
    rho_out = rho_null-diff

    def periodic_boundary_conditions(self):
        self.calcdensity()
        self.calcvelocity()
        equilibrium = self.calcequip(self.d, self.u)
        equilibrium_in = self.calcequip(self.rho_in, self.u[:,-2,:])
        self.f[:, 0, :] = equilibrium_in + (self.f[:, -2, :] - equilibrium[:, -2, :])
        equilibrium_out = self.calcequip(self.rho_out, self.u[:, 1, :])
        self.f[:, -1, :] = equilibrium_out + (self.f[:, 1, :] - equilibrium[:, 1, :])

    def calcequip(self,d,v):
        return ( ( ( 1 + 3*(np.dot(v.T,self.velocity_vector.T).T) + 9/2*((np.dot(v.T,self.velocity_vector.T) ** 2).T) - 3/2*(np.sum(v ** 2, axis=0)) ) * d ).T * self.weights ).T

    def simulation(self):
        shear_viscosity = (1/self.relaxation-0.5)/3
        self.d=np.ones((self.size_x+2,self.size_y+2))
        self.u=np.zeros((2,self.size_x+2,self.size_y+2))
        self.f=self.calcequi()
        for i in range(self.steps):
            self.periodic_boundary_conditions()
            self.stream()
            self.bounce_back_choosen()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            if (i % self.every) == 0 and i > 0:
                self.every = self.every*2
                plt.plot(self.u[0,50, 1:-1],np.arange(0,50),label = "Calculated {}".format(i))
        delta = 2.0 * self.diff /self.size_x / shear_viscosity / 2.
        y = np.linspace(0, self.size_y, self.size_y+1) + 0.5
        u_analytical = delta * y * (self.size_y - y) / 3.
        plt.plot(u_analytical[:-1],np.arange(0,50), label='Analytical')
        xmin,xmax,ymin,ymax=plt.axis()
        plt.plot([xmin,xmax],[ymin+2,ymin+2], color='k')
        plt.plot([xmin,xmax],[ymax-2,ymax-2], color='k')
        plt.legend()
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Pouisuelle flow (Gridsize 100x52, $\omega = 0.5$, $\\nabla \\rho = 0.002$)')
        savestring = "simulation_output/poiseuille/PouisuelleFlow.png"
        plt.savefig(savestring)
        plt.show()

p=poiseuille(size_x=100,size_y=50,steps=5000,every=10,velocity_lid=0.0,boundary_info=Boundary(False,False,True,True))
p.simulation()