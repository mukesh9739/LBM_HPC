import numpy as np
from dataclasses import dataclass
from base import Base
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import make_interp_spline

@dataclass
class shearwave(Base):
    # Perturbation magnitude for initial velocity and density profiles
    eps = 0.05
    
    def instant_theoretical_velocity(self, v: float) -> np.ndarray:
        """
        Calculate theoretical velocity decay for shear wave analysis.

        Parameters:
        - v (float): Kinematic viscosity.

        Returns:
        - Theoretical velocity decay as a numpy array.
        """
        return np.exp(-v * (2 * np.pi / self.size_y) ** 2)

    def decay_perturbation(self, t: float, viscosity: float) -> np.ndarray:
        """
        Calculate the decay of perturbation over time due to viscosity.

        Parameters:
        - t (float): Time step.
        - viscosity (float): Kinematic viscosity of the fluid.

        Returns:
        - Decay of perturbation as a function of time.
        """
        return self.eps * np.exp(-viscosity * (2 * np.pi / (self.size_y))**2 * t)

    def simulation_velocity(self, option: str = "None"):
        """
        Simulate velocity profiles for shear wave and generate plots for analysis.

        Parameters:
        - option (str): Determines the output of the function; primarily for internal use.

        Performs the simulation of velocity profile evolution due to shear wave perturbation,
        outputs graphical analysis, and saves the plots to disk.
        """
        # Initialize density and velocity matrices
        self.d = np.ones((self.size_y, self.size_x))
        self.u = np.zeros((2, self.size_y, self.size_x))
        # Apply initial velocity perturbation along one direction
        self.u[1, :, :] = self.eps * np.sin(2 * np.pi / self.size_y * np.arange(self.size_y)[:, np.newaxis])
        # Calculate the equilibrium distribution function
        self.f = self.calcequi()
        
        # Lists to hold simulation data for analysis
        min_vel_list, max_vel_list, theoretical_velocity, velocity_decay = [], [], [], []
        v_start = self.u
        
        # Simulation loop
        for step in range(self.steps):
            print(f'{step + 1}//{self.steps}', end="\r")
            self.stream()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            
            # Conditionally generate plots at specified intervals
            if(step % self.every == 0):
                plt.clf()
                X, Y = np.meshgrid(np.arange(self.size_y), np.arange(self.size_x))
                v_mag = np.sqrt(self.u[0]**2 + self.u[1]**2)
                plt.scatter(X, Y, c=v_mag, vmin=np.min(v_mag), vmax=np.max(v_mag))
                plt.title(f"Shear wave step {step}")
                plt.xlabel("x-Position")
                plt.ylabel("y-Position")
                plt.colorbar(label="Velocity")
                plt.savefig(f'simulation_output/shearwave/velocity/shear wave step {step}.png')
                
                # Store velocity profile for analysis
                velocity_decay.append(self.u[1, :, self.size_x // 4])
            
            # Update velocity profiles for analysis
            kinematic_viscosity = 1 / 3 * (1 / self.relaxation - 1 / 2)
            y_val = self.instant_theoretical_velocity(kinematic_viscosity)
            y_val *= v_start[1, self.size_x // 4, :]
            v_start[1, self.size_x // 4, :] = y_val
            theoretical_velocity.append(y_val.max())
            max_vel_list.append(np.max(self.u[1, :, :]))
            min_vel_list.append(np.min(self.u[1, :, :]))

        # Final analysis and plotting after simulation loop
        plt.clf()
        plt.ylim([-self.eps, self.eps])
        for data in velocity_decay:
            plt.plot(np.arange(self.size_y), data, alpha=0.5)
        plt.xlabel('Y Position')
        plt.ylabel(f'Velocity u(x = {self.size_x // 4}, y)')
        plt.grid()
        if option != "Return":
            plt.savefig(f'simulation_output/shearwave/velocity/velocity_decay_combined.png', pad_inches=1)
        plt.close()

        # Plot velocity maxima, minima and theoretical velocity over time
        x = np.arange(self.steps)
        plt.xlim([0, len(x)])
        X_Y_Spline = make_interp_spline(np.arange(self.steps), max_vel_list)
        X_Y_Spline_Min = make_interp_spline(np.arange(self.steps), min_vel_list)
        n = np.arange(self.steps)
        X_ = np.linspace(0, n[-1], 500)
        Y_ = X_Y_Spline(X_)
        Y_Min = X_Y_Spline_Min(X_)
        plt.plot(X_, Y_, color='blue')
        plt.plot(X_, Y_Min, color='blue')
        plt.plot(x, theoretical_velocity, color='black', linestyle='dotted', linewidth=3)
        plt.fill_between(X_, Y_, Y_Min, color="blue", alpha=.2)
        plt.xlabel('Time evolution')
        plt.ylabel(f'Velocity at (y = {self.size_y // 4})')
        plt.legend(['Simulated Maxima', 'Simulated Minima', 'Analytical ux(y=25)'])
        plt.grid()
        if option != "Return":
            plt.savefig(f'simulation_output/shearwave/velocity/omega_{self.relaxation}.png', pad_inches=1)
        plt.close()

        # Calculate simulated and analytical viscosity for comparison
        simulated_viscosity = curve_fit(self.decay_perturbation, xdata=x, ydata=max_vel_list)[0][0]
        analytical_viscosity = (1 / 3) * ((1 / self.relaxation) - 0.5)
        if option == "Return":
            return simulated_viscosity, analytical_viscosity
        else:
            print(simulated_viscosity)
            print(analytical_viscosity)

    def simulation_density(self):
        """
        Simulates density profiles for shear wave analysis and generates plots for visualization.
        """
        # Grid setup for plotting
        x, y = np.meshgrid(np.arange(self.size_x), np.arange(self.size_y))
        # Initialize density with a sinusoidal perturbation
        self.d = 1 + self.eps * np.sin(2 * np.pi / self.size_x * x)
        self.u = np.zeros((2, self.size_y, self.size_x), dtype=np.float32)
        self.f = self.calcequi()

        # Lists to store data for analysis
        dens = []
        den_decay = []

        # Simulation loop for density
        for step in range(self.steps):
            print(f'{step + 1}//{self.steps}', end="\r")
            self.stream()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            if(step % self.every == 0):
                # Plot and save density profile at specified intervals
                plt.clf()
                plt.scatter(x, y, c=self.d, vmin=np.min(self.d), vmax=np.max(self.d))
                plt.xlabel('X meshgrid')
                plt.ylabel('Y meshgrid')
                plt.title('Density flow in shear wave decay')
                plt.colorbar()
                plt.savefig(f'simulation_output/shearwave/density/density_{step}.png')
                den_decay.append(self.d[self.size_y // 4, :])

        # Additional plotting for density analysis after the simulation loop...
        # This part of the code will include analysis similar to the velocity simulation,
        # focusing on the decay of density perturbations and comparing them to theoretical predictions.
        
        # Note: The implementation for this part is not fully shown in the initial snippet
        # and would involve plotting the density decay over time and analyzing its behavior.

    def kinemvsomega(self):
        """
        Analyzes the relationship between kinematic viscosity and the relaxation parameter omega.
        """
        # Range of omega values for analysis
        omega_values = np.arange(0.1, 2, 0.2)
        sim_viscosities = []
        analytical_viscosities = []

        # Loop over omega values to simulate and calculate viscosity
        for omega in omega_values:
            self.relaxation = omega
            simulated_viscosity, analytical_viscosity = self.simulation_velocity(option="Return")
            sim_viscosities.append(simulated_viscosity)
            analytical_viscosities.append(analytical_viscosity)

        # Plotting the relationship between kinematic viscosity and omega
        plt.figure(figsize=(8, 6))
        plt.plot(omega_values, sim_viscosities, 'r-', label='Simulated Viscosity')
        plt.plot(omega_values, analytical_viscosities, 'k--', label='Analytical Viscosity')
        plt.title('Kinematic Viscosity vs Omega')
        plt.xlabel('Omega')
        plt.ylabel('Kinematic Viscosity')
        plt.legend()
        plt.grid(True)
        plt.savefig('simulation_output/shearwave/kinematic_viscosity_vs_omega.png')
        plt.show()

# Instantiate the shearwave class with specified lattice dimensions, steps, and update frequency
c = shearwave(size_x=150, size_y=200, steps=25000, every=3000, relaxation=1)

# Execute the simulation for velocity and visualize the results
c.simulation_velocity()

# Execute the simulation for density and visualize the results
c.simulation_density()

# Analyze the relationship between kinematic viscosity and omega
c.kinemvsomega()
