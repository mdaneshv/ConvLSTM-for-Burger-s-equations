import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  


def lorenz(x, y, z, sigma=10, rho=28, beta=2.667):
    '''
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
       Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    dx = sigma*(y - x)
    dy = rho*x - y - x*z
    dz = x*y - beta*z
    return dx, dy, dz


dt = 0.0005
num_steps = 600000

# Need one more for the initial values
xs = np.zeros(num_steps + 1)
ys = np.zeros(num_steps + 1)
zs = np.zeros(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0, 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    dx, dy, dz = lorenz(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (dx * dt)
    ys[i + 1] = ys[i] + (dy * dt)
    zs[i + 1] = zs[i] + (dz * dt)


# Plot
fig = plt.figure()
ax = fig.gca()

ax.plot(xs)
ax.set_xlabel("X Axis")
#ax.set_ylabel("Y Axis")
#ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
print(xs[600000])
np.savetxt('xs.csv',xs,delimiter=',')
#ax.plot(xs)
#ax.set_xlabel("X Axis")
#ax.set_ylabel("Y Axis")
#ax.set_zlabel("Z Axis")
#ax.set_title("Lorenz Attractor")

#plt.show()
