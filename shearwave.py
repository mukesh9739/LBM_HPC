import numpy as np
from dataclasses import dataclass
from base import Base
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from scipy.interpolate import make_interp_spline

@dataclass
class shearwave(Base):
    eps=0.05
    
    def instant_theoretical_velocity(self,v):
        return np.exp(-v*(2*np.pi/self.size_y) ** 2)

    def decay_perturbation(self, t, viscosity):
        return self.eps * np.exp(-viscosity * (2*np.pi/(self.size_y))**2 * t)

    def simulation_velocity(self,option="None"):
        self.d=np.ones((self.size_y,self.size_x))
        self.u=np.zeros((2,self.size_y,self.size_x))
        self.u[1, :, :] =  self.eps * np.sin(2*np.pi/self.size_y*np.arange(self.size_y)[:, np.newaxis])
        self.f=self.calcequi()
        min_vel_list, max_vel_list, theoretical_velocity, velocity_decay =[], [], [], []
        v_start=self.u
        for step in range(self.steps):
            print(f'{step+1}//{self.steps}', end="\r")
            self.stream()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            if(step%self.every==0):
                plt.clf()
                X, Y = np.meshgrid(np.arange(self.size_y-2), np.arange(self.size_x-2))
                v_mag = np.sqrt(self.u[0, 1:-1, 1:-1]**2 + self.u[1, 1:-1, 1:-1]**2)
                plt.scatter(X, Y, c=v_mag, vmin=np.min(v_mag), vmax=np.max(v_mag))
                plt.title("Shear wave step".format(step))
                plt.xlabel("x-Position")
                plt.ylabel("y-Position")
                fig = plt.colorbar()
                fig.set_label("Velocity", rotation=270, labelpad=15)
                plt.savefig(f'simulation_output/shearwave/velocity/shear wave step {step}')                
            kinematic_viscosity = 1/3*(1/self.relaxation-1/2)
            y_val = self.instant_theoretical_velocity(kinematic_viscosity)
            y_val = y_val*(v_start[1, self.size_x//4, :])
            v_start[1, self.size_x//4, :] = y_val
            theoretical_velocity.append(y_val.max())
            max_vel_list.append(np.max(self.u[1, :, :]))
            min_vel_list.append(np.min(self.u[1, :, :]))
            if step % self.every == 0:
                plt.ylim([-self.eps, self.eps])
                plt.plot(np.arange(self.size_y), self.u[1, :, self.size_x // 4])
                plt.xlabel('Y Position')
                plt.ylabel(f'Velocity u(x = {self.size_x // 4},y)')
                plt.grid()
                velocity_decay.append(self.u[1, :, self.size_x // 4])
        plt.clf()
        plt.ylim([-self.eps, self.eps])
        for data in velocity_decay:
            plt.plot(np.arange(self.size_y), data, alpha=0.5)
        plt.xlabel('Y Position')
        plt.ylabel(f'Velocity u(x = {self.size_x // 4},y)')
        plt.grid()
        if option != "Return":
            plt.savefig(f'simulation_output/shearwave/velocity/velocity_decay_combined.png', pad_inches=1)
        plt.close()
        x = np.arange(self.steps)
        plt.xlim([0, len(x)])
        X_Y_Sp = make_interp_spline(np.arange(self.steps), max_vel_list)
        X_Y_Sp_1 = make_interp_spline(np.arange(self.steps), min_vel_list)
        n=np.arange(self.steps)
        X_ = np.linspace(0, n[-1], 500)
        Y_ = X_Y_Sp(X_)
        Y_1 = X_Y_Sp_1(X_)
        plt.plot(X_, Y_, color='blue')
        plt.plot(X_, Y_1, color='blue')
        plt.plot(x, theoretical_velocity, color='black', linestyle='dotted', linewidth=3)
        plt.fill_between(X_, Y_, Y_1, color="blue", alpha=.2)
        plt.xlabel('Time evolution')
        plt.ylabel(f'Velocity at (y = {self.size_y // 4})')
        plt.legend(['Simulated Maxima', 'Simulated Minima', 'Analytical ux(y=25)'])
        plt.grid()
        if option != "Return":
            plt.savefig(f'simulation_output/shearwave/velocity/omega_{self.relaxation}.png', pad_inches=1)
        plt.close()
        simulated_viscosity = curve_fit(self.decay_perturbation, xdata=x, ydata=max_vel_list)[0][0]
        analytical_viscosity = (1 / 3) * ((1 / self.relaxation) - 0.5)
        if option == "Return":
            return simulated_viscosity, analytical_viscosity
        else:
            print(simulated_viscosity)
            print(analytical_viscosity)

    def simulation_density(self):
        x, y = np.meshgrid(np.arange(self.size_x), np.arange(self.size_y))
        self.d = 1 + self.eps * np.sin(2*np.pi/self.size_x*x)
        self.u = np.zeros((2,self.size_y,self.size_x), dtype=np.float32)
        self.f = self.calcequi()
        den_list, max_min_list,dens =[], [], []
        for step in range(self.steps):
            print(f'{step+1}//{self.steps}', end="\r")
            self.stream()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            if(step%self.every==0):
                plt.clf()
                plt.scatter(x, y, c=self.d, vmin=np.max(self.d), vmax=np.max(self.d))
                plt.xlabel('X meshgrid')
                plt.ylabel('Y meshgrid')
                plt.title('Density flow in shear wave decay')
                plt.colorbar()
                plt.savefig(f'simulation_output/shearwave/density/density_{step}.png')
                plt.cla()
                plt.ylim([-self.eps + 1, self.eps + 1])
                plt.plot(np.arange(self.size_x), self.d[self.size_y // 4, :])
                plt.xlabel('X Position')
                plt.ylabel(f'Density ρ (y = {self.size_y // 4})')
                plt.grid() 
                plt.savefig(f'simulation_output/shearwave/density/density_decay_{step}.png')
            dens.append(self.d[self.size_y // 4, self.size_x // 4] - 1)
            den_list.append(np.max(self.d - 1))
            max_min_list.append(np.min(self.d - 1))
        den_list = np.array(den_list)
        xl = argrelextrema(den_list, np.greater)[0]
        den_list = den_list[xl]
        xt = np.arange(len(dens))
        plt.figure()
        plt.plot(xt, dens, color='blue', label='Simulated Density')
        plt.xlabel(f'Timestep (ω = {self.relaxation})')
        plt.ylabel(f'Density ρ (x = {self.size_x // 4}, y = {self.size_y // 4})')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'simulation_output/shearwave/density/omegadd_{self.relaxation}.png', bbox_inches='tight', pad_inches=0.5)
        plt.show()

    def kinemvsomega(self):
        ov=np.arange(0.1,2,0.2)
        sim_ves=[]
        ana_ves=[]
        for i in ov:
            self.relaxation=i
            a,b=self.simulation_velocity(option="Return")
            sim_ves.append(a)
            ana_ves.append(b)
        plt.plot(ov,sim_ves,color='red', label='Simulated')
        plt.plot(ov,ana_ves,color='black',linestyle = 'dotted', linewidth= 2,label='Calculated')
        plt.title("Kinematic viscosity vs Omega")
        plt.xlabel("Omega")
        plt.ylabel("Kinematic Viscosity")
        plt.legend()
        plt.grid()
        plt.savefig('simulation_output/shearwave/Kinematic viscosity vs Omega 1000')
        plt.show()

c=shearwave(size_x=100,size_y=100,steps=20000,every=1000,relaxation=1)
c.simulation_velocity()
c.simulation_density()
c.kinemvsomega()