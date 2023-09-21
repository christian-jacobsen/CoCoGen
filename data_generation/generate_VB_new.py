import numpy as np
import matplotlib.pyplot as plt

def viscous_burgers(viscosity, num_points, num_steps, dt):
    # Define the x-grid and initialize the solution
    x = np.linspace(0, 1, num_points)
    dx = x[1] - x[0]
    u = np.sin(2 * np.pi * x)

    # Define the RK4 time-stepping scheme
    def rk4_step(u, f, dt):
        k1 = dt * f(u)
        k2 = dt * f(u + 0.5 * k1)
        k3 = dt * f(u + 0.5 * k2)
        k4 = dt * f(u + k3)
        return u + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Define the RHS of the Burgers equation
    def f(u):
        u_xx = np.roll(u, 1) - 2 * u + np.roll(u, -1)
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2*dx)
        return -u * u_x + viscosity * u_xx / dx**2

    # Time-step the solution using RK4
    u_mat = np.zeros((num_points, num_steps))
    for n in range(num_steps):
        u = rk4_step(u, f, dt)
        u_mat[:, n] = u

    return x, u_mat

def main():
    n_samples = 1000 # number of data samples
    n_x = 64 # number of discretization points on spatial domain [0, 1]
    dt = 1e-2 # time step
    n_t = 256 # number of steps

    x, u =  viscous_burgers(0.01, n_x, n_t, dt)

    plt.figure(1)
    plt.imshow(u)
    plt.savefig('./viz_burg.png')



    # for i in range(n_samples):





if __name__ == '__main__':
    main()

