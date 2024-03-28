import matplotlib.pyplot as plt
import numpy as np

# Let's add all the provided data and generate the plot

# Data extracted and processed for plotting
data = [
    {"lattice": 200, "decomp": 2, "mlups": 8333333},
    {"lattice": 200, "decomp": 4, "mlups": 23529411},
    {"lattice": 200, "decomp": 5, "mlups": 26666666},
    {"lattice": 200, "decomp": 8, "mlups": 20000000},
    {"lattice": 200, "decomp": 10, "mlups": 23529411},
    {"lattice": 200, "decomp": 20, "mlups": 25000000},
    {"lattice": 200, "decomp": 25, "mlups": 25000000},
    {"lattice": 200, "decomp": 40, "mlups": 25000000},
    {"lattice": 200, "decomp": 50, "mlups": 25000000},
    {"lattice": 400, "decomp": 2, "mlups": 9937888},
    {"lattice": 400, "decomp": 4, "mlups": 36363636},
    {"lattice": 400, "decomp": 5, "mlups": 45714285},
    {"lattice": 400, "decomp": 8, "mlups": 51612903},
    {"lattice": 400, "decomp": 10, "mlups": 59259259},
    {"lattice": 400, "decomp": 16, "mlups": 80000000},
    {"lattice": 400, "decomp": 20, "mlups": 84210526},
    {"lattice": 400, "decomp": 25, "mlups": 94117647},
    {"lattice": 400, "decomp": 40, "mlups": 100000000},
    {"lattice": 400, "decomp": 50, "mlups": 100000000},
    {"lattice": 600, "decomp": 2, "mlups": 9498680},
    {"lattice": 600, "decomp": 3, "mlups": 18848167},
    {"lattice": 600, "decomp": 4, "mlups": 37113402},
    {"lattice": 600, "decomp": 5, "mlups": 52941176},
    {"lattice": 600, "decomp": 6, "mlups": 66666666},
    {"lattice": 600, "decomp": 8, "mlups": 80000000},
    {"lattice": 600, "decomp": 10, "mlups": 100000000},
    {"lattice": 600, "decomp": 12, "mlups": 109090909},
    {"lattice": 600, "decomp": 15, "mlups": 124137931},
    {"lattice": 600, "decomp": 20, "mlups": 150000000},
    {"lattice": 600, "decomp": 24, "mlups": 163636363},
    {"lattice": 600, "decomp": 25, "mlups": 171428571},
    {"lattice": 600, "decomp": 30, "mlups": 189473684},
    {"lattice": 600, "decomp": 40, "mlups": 200000000},
    {"lattice": 600, "decomp": 50, "mlups": 211764705},
    {"lattice": 800, "decomp": 2, "mlups": 8803301},
    {"lattice": 800, "decomp": 4, "mlups": 31067961},
    {"lattice": 800, "decomp": 5, "mlups": 48484848},
    {"lattice": 800, "decomp": 8, "mlups": 90140845},
    {"lattice": 800, "decomp": 10, "mlups": 112280701},
    {"lattice": 800, "decomp": 16, "mlups": 188235294},
    {"lattice": 800, "decomp": 20, "mlups": 213333333},
    {"lattice": 800, "decomp": 25, "mlups": 256000000},
    {"lattice": 800, "decomp": 32, "mlups": 290909090},
    {"lattice": 800, "decomp": 40, "mlups": 320000000},
    {"lattice": 800, "decomp": 50, "mlups": 336842105},
    {"lattice": 1000, "decomp": 2, "mlups": 8417508},
    {"lattice": 1000, "decomp": 4, "mlups": 29154518},
    {"lattice": 1000, "decomp": 5, "mlups": 43103448},
    {"lattice": 1000, "decomp": 8, "mlups": 88495575},
    {"lattice": 1000, "decomp": 10, "mlups": 138888888},
    {"lattice": 1000, "decomp": 20, "mlups": 285714285},
    {"lattice": 1000, "decomp": 25, "mlups": 333333333},
    {"lattice": 1000, "decomp": 40, "mlups": 416666666},
    {"lattice": 1000, "decomp": 50, "mlups": 476190476},
]

# Plot
fig, ax = plt.subplots()
for lattice in set(item['lattice'] for item in data):
    # Filtering data for each lattice size
    filtered_data = [item for item in data if item['lattice'] == lattice]
    decomps = [item['decomp'] for item in filtered_data]
    mlups = [item['mlups'] for item in filtered_data]
    processors = [d**2 for d in decomps]  # Calculate the number of processors

    ax.plot(processors, mlups, 'o-', label=f'Lattice {lattice}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of Processors (= decomp^2)')
ax.set_ylabel('MLUPS')
ax.set_title('Logarithmic Graph of MLUPS vs Number of Processors')
ax.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("mlups.png")
plt.show()
