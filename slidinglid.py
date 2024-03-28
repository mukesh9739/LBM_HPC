import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class slidinglid(Base):
    """
    Class for simulating fluid flow in a lid-driven cavity using the Lattice Boltzmann Method (LBM).
    Inherits from the Base class to utilize core LBM functionalities.
    """

    def plotter(self, i):
        """
        Generates and saves a plot of the fluid velocity field at a given timestep.
        
        Parameters:
        - i: The current timestep being plotted.
        """
        # Clear the current figure
        plt.clf()
        # Define the grid for plotting
        x = np.arange(0, self.size_x - 2)
        y = np.arange(0, self.size_y - 2)
        X, Y = np.meshgrid(x, y)
        # Calculate the speed at each point for color coding
        speed = np.sqrt(self.u[0, 2:-2, 2:-2].T ** 2 + self.u[1, 2:-2, 2:-2].T ** 2)
        # Create a streamplot of the velocity field
        plt.streamplot(X, Y, self.u[0, 2:-2, 2:-2].T, self.u[1, 2:-2, 2:-2].T, color=speed, cmap=plt.cm.jet)
        # Set the plot limits
        ax = plt.gca()
        ax.set_xlim([0, self.size_x + 1])
        ax.set_ylim([0, self.size_y + 1])
        # Title with simulation parameters
        titleString = f"Sliding Lid (Gridsize {self.size_x}x{self.size_y}, $\\omega$ = {self.relaxation:.2f}, steps = {i})"
        plt.title(titleString)
        plt.xlabel("x-Position")
        plt.ylabel("y-Position")
        # Add a color bar indicating speed
        fig = plt.colorbar()
        fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
        # Save the plot to a file
        savestring = f"simulation_output/slidinglid/slidinglid_step_{i}.png"
        plt.savefig(savestring)

    def simulation(self):
        """
        Executes the sliding lid cavity simulation, periodically invoking the plotter to generate velocity field plots.
        """
        print(self.relaxation)
        # Initialize density and velocity fields
        self.d = np.ones((self.size_x + 2, self.size_y + 2))
        self.u = np.zeros((2, self.size_x + 2, self.size_y + 2))
        # Compute the initial equilibrium distribution
        self.f = self.calcequi()
        
        # Simulation loop
        for i in range(self.steps):
            self.stream()  # Perform streaming step
            self.bounce_back_choosen()  # Apply boundary conditions
            self.calcdensity()  # Calculate density
            self.calcvelocity()  # Calculate velocity
            self.calccollision()  # Perform collision step
            
            # Plot the velocity field at specified intervals
            if (i % self.every) == 0 and i > 0:
                self.every *= 2  # Adjust plotting interval
                self.plotter(i)  # Generate plot for the current timestep
                
        # Generate a final plot at the end of the simulation
        self.plotter(self.steps)

# Parameters for the simulation, including Reynolds number and domain size
re = 1000
base_length = 150
uw = 0.1  # Velocity of the sliding lid

# Instantiate and run the slidinglid simulation
sl = slidinglid(size_x=base_length, size_y=base_length, every=10000000, steps=10, 
                relaxation=((2 * re) / (6 * base_length * uw + re)), velocity_lid=uw, 
                boundary_info=Boundary(True, True, True, True))
sl.simulation()
