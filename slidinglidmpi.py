import numpy as np
from dataclasses import dataclass
from mpi4py import MPI
import time
import sys
from base import Base
import matplotlib.pyplot as plt

startime = time.time()  # Record the start time of the simulation

@dataclass
class SlidingLidMPI(Base):
    """
    Extends the Base class to implement the sliding lid cavity problem using parallel computing with MPI.
    """
    # Position in the MPI grid
    pos_x: int = -1
    pos_y: int = -1
    # MPI rank and size
    rank: int = -1
    size: int = -1
    # Base grid size
    base_grid: int = -1
    # MPI communicator
    mpi: MPI = None

    def get_postions_out_of_rank_size_quadratic(self, rank, size):
        """
        Calculates the grid position based on the MPI rank and total size.
        """
        return (rank % int(np.sqrt(size))), (rank // int(np.sqrt(size)))

    def fill_mpi_struct_fields(self):
        """
        Initializes the simulation fields and sets boundary and neighbor information for MPI processes.
        """
        # Determine the position in the MPI grid
        self.pos_x, self.pos_y = self.get_postions_out_of_rank_size_quadratic(self.rank, self.size)
        # Set boundary info based on the position
        self.boundary_info.set_boundary_info(self.pos_x, self.pos_y, int(np.sqrt(self.size)) - 1, int(np.sqrt(self.size)) - 1)
        # Determine neighbors in the MPI grid
        self.neighbours_info.determin_neighbors(self.rank, self.size)
        # Initialize density and velocity fields
        self.d = np.ones((self.size_x, self.size_y))
        self.u = np.zeros((2, self.size_x, self.size_y))
        # Calculate initial equilibrium distribution function
        self.f = self.calcequi()

    def sliding_lid_mpi(self):
        """
        Performs the sliding lid simulation steps and aggregates data across MPI processes.
        """
        for i in range(self.steps):
            self.stream()
            self.bounce_back_choosen()
            self.calcdensity()
            self.calcvelocity()
            self.calccollision()
            # Communicate boundary data with neighboring processes
            self.comunicate()
        # Aggregate data to create a full simulation grid
        full_grid = self.collapse_data()
        # If this is the master process, visualize the result
        if self.rank == 0:
            self.plotter(full_grid)

    def collapse_data(self):
        """
        Aggregates data from all MPI processes to assemble the full simulation grid on the master process.
        """
        full_grid = np.ones((9, self.base_grid, self.base_grid))
        if self.rank == 0:
            # Master process gathers data from all other processes
            full_grid[:, 0:self.size_x - 2, 0:self.size_y - 2] = self.f[:, 1:-1, 1:-1]
            temp = np.zeros((9, self.size_x - 2, self.size_y - 2))
            for i in range(1, self.size):
                self.mpi.Recv(temp, source=i, tag=i)
                x, y = self.get_postions_out_of_rank_size_quadratic(i, self.size)
                full_grid[:, (0 + (self.size_x - 2) * x):((self.size_x - 2) + (self.size_x - 2) * x), (0 + (self.size_y - 2) * y):((self.size_y - 2) + (self.size_y - 2) * y)] = temp
        else:
            # Other processes send their data to the master process
            self.mpi.Send(self.f[:, 1:-1, 1:-1].copy(), dest=0, tag=self.rank)
        return full_grid

    def comunicate(self):
        """
        Handles the communication of boundary data between neighboring MPI processes to ensure data consistency across process boundaries.
        """
        # Exchange boundary data with neighboring processes based on the current boundary configuration
        # This section involves sending and receiving boundary rows/columns with adjacent processes
        # to synchronize the overlapping boundary regions in the simulation grid.
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
           
    def plotter(self, full_grid):
        """
        Visualizes the velocity field for the aggregated simulation grid on the master process.
        """
        if self.rank == 0:
            print("Making Image")
            self.f = full_grid
            # Calculate density and velocity for visualization
            self.calcdensity()
            self.calcvelocity()
            # Prepare the velocity field for plotting
            x = np.arange(0, self.base_grid)
            y = np.arange(0, self.base_grid)
            X, Y = np.meshgrid(x, y)
            speed = np.sqrt(self.u[0].T ** 2 + self.u[1].T ** 2)  # Calculate speed for color coding
            plt.streamplot(X, Y, self.u[0].T, self.u[1].T, color=speed, cmap=plt.cm.jet)  # Create a stream plot
            ax = plt.gca()
            ax.set_xlim([0, self.base_grid + 1])
            ax.set_ylim([0, self.base_grid + 1])
            plt.title("Sliding Lid")
            plt.xlabel("x-Position")
            plt.ylabel("y-Position")
            fig = plt.colorbar()  # Add a color bar to indicate speed
            fig.set_label("Velocity u(x,y,t)", rotation=270, labelpad=15)
            savestring = "slidingLidmpi" + str(self.size) + ".png"  # Save the plot to a file
            plt.savefig(savestring)
            plt.show()

def call():
    # Entry point for the MPI sliding lid simulation
    if len(sys.argv) < 3:
        print("Usage: python script.py [base_length] [steps]")
        sys.exit(1)
        
    steps = int(sys.argv[2])
    re = 1000  # Reynolds number
    base_length = int(sys.argv[1])
    uw = 0.1  # Velocity of the sliding lid
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank_1d = int(np.sqrt(size))  # Assume a square MPI process grid
    
    if rank_1d * rank_1d != size:
        print("Error: The number of MPI processes must be a perfect square.")
        sys.exit(1)
        
    # Initialize the sliding lid simulation with MPI-specific parameters
    slmpi = SlidingLidMPI(
        rank=comm.Get_rank(),
        size=comm.Get_size(),
        size_x=base_length // rank_1d + 2,  # Adjust grid size per MPI process
        size_y=base_length // rank_1d + 2,
        relaxation=((2 * re) / (6 * base_length * uw + re)),  # Calculate relaxation parameter
        steps=steps,
        velocity_lid=uw,
        base_grid=base_length,
        mpi=comm
    )
    slmpi.fill_mpi_struct_fields()  # Initialize simulation fields
    slmpi.sliding_lid_mpi()  # Run the simulation

call()