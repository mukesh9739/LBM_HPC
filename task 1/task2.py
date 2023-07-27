import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

x=3
y=2
r=0.5
c = np.array([[ 0, 1, 0,-1, 0, 1,-1,-1, 1],[ 0, 0, 1, 0,-1, 1, 1,-1,-1]]).T
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]).T
f=np.random.rand(x,y,9)
d=np.zeros((x,y))
v=np.zeros((x,y,2))

for i in range(x):
    for k in range(y):
        f[i,k,:]=np.random.uniform(0,1,size=(1,9))#dirichlet(np.ones(9),size=1)

def calcdensity(f):
    return f.sum(axis=2)

def calcvelocity(f,d):
    return (np.dot(f, c).T / d.T).T 

def calccollision(f, relaxation):  
    d = calcdensity(f)    
    v = calcvelocity(f,d)
    feq = calcequi(d,v)
    f -= relaxation * (f-feq)    
    return f, d, v

def calcequi(d, v):
    vel_mag = v[:,:,0] ** 2 + v[:,:,1] ** 2
    cu = np.dot(v,c.T)
    sq_velocity = cu ** 2
    f_eq = ((1 + 3*(cu.T) + 9/2*(sq_velocity.T) - 3/2*(vel_mag.T)) * d.T ).T * w
    return f_eq

def stream(f):   
    for i in range(9):
        f[:,:,i] = np.roll(f[:,:,i],c[i], axis = (0,1)) 
    global d,v
    f,d,v=calccollision(f,r)        
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

anim = animation.FuncAnimation(fig, animate, frames=100, interval=10, blit=False, repeat=False)
anim.save('./animation1.gif', writer='imagemagick', fps=60)
