import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class poiseuille(Base):
    """
    Simulates Poiseuille flow using the Lattice Boltzmann Method (LBM).
    Inherits functionalities from the Base class to model fluid flow through a channel.
    """
    # Base density for the fluid
    rho_null = 1
    # Pressure difference across the channel
    diff=0.001
    # Inlet and outlet pressures for driving the flow
    rho_in = rho_null+diff
    rho_out = rho_null-diff

    def periodic_boundary_conditions(self):
        """
        Applies periodic boundary conditions to simulate Poiseuille flow,
        maintaining a constant pressure gradient across the channel.
        """
        self.calcdensity()
        self.calcvelocity()
        # Calculate equilibrium distribution for the bulk
        equilibrium = self.calcequip(self.d, self.u)
        # Inlet equilibrium distribution
        equilibrium_in = self.calcequip(self.rho_in, self.u[:,-2,:])
        # Adjust inlet distribution to maintain pressure difference
        self.f[:, 0, :] = equilibrium_in + (self.f[:, -2, :] - equilibrium[:, -2, :])
        # Outlet equilibrium distribution
        equilibrium_out = self.calcequip(self.rho_out, self.u[:, 1, :])
        # Adjust outlet distribution to maintain pressure difference
        self.f[:, -1, :] = equilibrium_out + (self.f[:, 1, :] - equilibrium[:, 1, :])

    def calcequip(self,d,v):
        """
        Calculates the equilibrium distribution function for given density and velocity.
        
        Parameters:
        - d: Density at each lattice point.
        - v: Velocity at each lattice point.
        
        Returns:
        - The equilibrium distribution function as a numpy array.
        """
        return ( ( ( 1 + 3*(np.dot(v.T,self.velocity_vector.T).T) + 9/2*((np.dot(v.T,self.velocity_vector.T) ** 2).T) - 3/2*(np.sum(v ** 2, axis=0)) ) * d ).T * self.weights ).T

    def simulation(self):
        """
        Executes the Poiseuille flow simulation, plotting the velocity profile
        and comparing it with the analytical solution.
        """
        # Calculate shear viscosity based on relaxation time
        shear_viscosity = (1/self.relaxation-0.5)/3
        # Initialize density and velocity fields
        self.d=np.ones((self.size_x+2,self.size_y+2))
        self.u=np.zeros((2,self.size_x+2,self.size_y+2))
        self.f=self.calcequi()

        for i in range(self.steps):
            # Apply periodic boundary conditions and perform LBM steps
            self.periodic_boundary_conditions()
            self.stream()
            self.bounce_back_choosen()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()

            if (i % self.every) == 0 and i > 0:
                # Plot the velocity profile at specified intervals
                self.every = self.every*2 # Dynamically adjust the plotting interval
                plt.plot(self.u[0,50, 1:-1],np.arange(0,50))
        
        # Final comparison of simulated and analytical velocity profiles
        plt.plot(self.u[0,50, 1:-1],np.arange(0,50),label = "simulated")
        # Analytical solution for Poiseuille flow
        delta = 2.0 * self.diff /self.size_x / shear_viscosity / 2.
        y = np.linspace(0, self.size_y, self.size_y+1) + 0.5
        u_analytical = delta * y * (self.size_y - y) / 3.
        plt.plot(u_analytical[:-1],np.arange(0,50), label='Analytical')

        # Visual adjustments and plot saving
        xmin,xmax,ymin,ymax=plt.axis()
        plt.plot([xmin,xmax],[ymin+2,ymin+2], color='k' , label = "Fixed Wall")
        plt.plot([xmin,xmax],[ymax-2,ymax-2], color='k')
        plt.legend(loc='lower right')
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Pouisuelle flow (Gridsize 100x52, $\omega = 0.5$, $\\nabla \\rho = 0.002$)')
        # Save the plot to a specified location
        savestring = "simulation_output/poiseuille/PouisuelleFlow.png"
        plt.savefig(savestring)
        # Display the plot
        plt.show()

# Create an instance of the poiseuille class with specified parameters
# including the size of the simulation domain, the number of steps, 
# the frequency of updates, and the velocity of the lid (which is zero for Poiseuille flow)
p=poiseuille(size_x=100,size_y=50,steps=2500,every=10,velocity_lid=0.0,boundary_info=Boundary(False,False,True,True))

# Run the Poiseuille flow simulation
p.simulation()