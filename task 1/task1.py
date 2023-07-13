import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

x=3
y=2
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
f=np.random.rand(x,y,9)
for i in range(x):
    for k in range(y):
        f[i,k,:]=np.random.dirichlet(np.ones(9),size=1)

def calcdensity(f):
    return f.sum(axis=2)

def calcvelocity(f,d):
    return (np.dot(f, c).T / d.T).T 

d = calcdensity(f)
v = calcvelocity(f,d)

def stream(f):   
    for i in range(9):
        f[:,:,i] = np.roll(f[:,:,i],c[i], axis = (0,1)) 
    global d,v
    d=calcdensity(f)
    v=calcvelocity(f,d)        
    return f

f=stream(f)

X, Y = np.meshgrid(np.arange(y),np.arange(x))
V= np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)

fig, ax = plt.subplots()
st = ax.streamplot(X,Y,d, V, color=d, density=2, cmap='jet',arrowsize=1)
fig.colorbar(st.lines)
plt.tight_layout()
plt.show()

def animate(i):
    ax.clear() # clear arrowheads streamplot
    global f,d,V
    f=stream(f)
    V= np.sqrt(v[:, :, 0]**2 + v[:, :, 1]**2)
    st = ax.streamplot(X,Y,d, V, color=d, density=2, cmap='jet',arrowsize=1)
    return st

anim =   animation.FuncAnimation(fig, animate, frames=100, interval=10, blit=False, repeat=False)
anim.save('./animation.gif', writer='imagemagick', fps=60)
#plt.show()





