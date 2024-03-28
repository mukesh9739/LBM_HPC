import numpy as np
from dataclasses import dataclass
from base import Base
from boundary import Boundary
import matplotlib.pyplot as plt

@dataclass
class couette(Base):
    """
    Simulates Couette flow using the Lattice Boltzmann Method (LBM).
    Inherits from Base class to utilize LBM core functionalities.
    """

    def simulation(self):
        """
        Runs the simulation for Couette flow, plots the velocity profile,
        and compares it with the analytical solution.
        """
        # Initialize density and velocity fields
        self.d = np.ones((self.size_x, self.size_y + 2))
        self.u = np.zeros((2, self.size_x, self.size_y + 2))
        self.f = self.calcequi()  # Calculate initial equilibrium distribution

        for i in range(self.steps):
            self.calcdensity()  # Calculate density
            self.calcvelocity()  # Calculate velocity
            self.calccollision()  # Perform collision step
            self.stream()  # Perform streaming step
            self.bounce_back_choosen()  # Apply bounce-back boundary conditions

            # Plot intermediate velocity profiles at specified intervals
            if (i % self.every) == 0 and i > 0:
                self.every = self.every * 2  # Adjust plot interval
                plt.plot(self.u[0, 50, 1:-1], np.arange(0, 50))  # Plot simulated velocity profile

        # Final plot of simulated velocity profile for the last timestep
        plt.plot(self.u[0, 50, 1:-1], np.arange(0, 50), label="Simulated")
        
        # Calculate and plot the analytical solution for comparison
        y = self.velocity_lid * 1 / 50 * np.arange(0, 50)
        plt.plot(y, np.arange(0, 50), label="Analytical")

        # Add labels and legend to the plot
        plt.xlabel('Position in cross section')
        plt.ylabel('Velocity')
        plt.title('Couette flow (Gridsize 100x50, $\omega = 0.5$)')
        
        # Highlight the fixed and moving walls in the plot
        xmin, xmax, ymin, ymax = plt.axis()
        plt.plot([xmin, xmax], [ymin + 2, ymin + 2], color='k', label="Fixed wall")
        plt.plot([xmin, xmax], [ymax - 2, ymax - 2], color='r', label="Moving wall")
        plt.legend()

        # Save the final plot to file and display it
        plt.savefig("simulation_output/couette/CouetteFlow.png")
        plt.show()

# Instantiate the couette class with simulation parameters and run the simulation
c = couette(size_x=100, size_y=50, steps=1950, every=100, velocity_lid=0.1, boundary_info=Boundary(False, False, True, True))
c.simulation()
