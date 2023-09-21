"""
Parallelized script to generate 2D Darcy flow data with parameterized permeability fields
Author: Christian Jacobsen, University of Michigan 2023
"""


import os
import os.path as osp
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import scipy
import argparse
import matplotlib.pyplot as plt
import time


def parse_args():
    parser = argparse.ArgumentParser(description='2D Darcy flow solver')
    parser.add_argument('-n', type=int, default=128, help='parameterization dimension of permeability field')
    parser.add_argument('--resolution', type=int, default=64, help='resultion of physical field; [0,1] in x and y by --resolution steps')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--threads', type=int, default=32, help='Number of cores to use during generation')
    parser.add_argument('--output', type=str, help='Path to output folder for data')
    parser.add_argument('--resume', type=int, default=0, help='Resume generation from sample i')
    return parser.parse_args()


def create_output_dirs(args, resume=False):
    if not osp.exists(args.output):
        os.makedirs(args.output)
    if not osp.exists(osp.join(args.output, "params")):
        os.makedirs(osp.join(args.output, "params"))
    if not osp.exists(osp.join(args.output, "permeability")):
        os.makedirs(osp.join(args.output, "permeability"))
    if not osp.exists(osp.join(args.output, "data")):
        os.makedirs(osp.join(args.output, "data"))
    elif not resume:
        raise Exception('data already exists')
    if not osp.exists(osp.join(args.output, "imgs")):
        os.makedirs(osp.join(args.output, "imgs"))


def enforce_integral(x, dx):
    nx = len(x)
    A = np.zeros((1, nx**2))
    j = 0
    i = 0
    for k in range(nx**2):
        if np.mod(k, nx)==0:
            j += 1
            i = 1

        if (i == 1) and (j == 1): # bottom left corner
            A[0, k] = 1
        elif (i == nx) and (j == nx): # top right corner
            A[0, k] = 1
        elif (i == 1) and (j == nx): # top left corner
            A[0, k] = 1
        elif (i == nx) and (j == 1): # bottom right corner
            A[0, k] = 1
        elif (i == 1): # left boundary
            A[0, k] = 2
        elif (j == 1): # bottom boundary
            A[0, k] = 2
        elif (i == nx): # right boundary
            A[0, k] = 2
        elif (j == nx): # top boundary
            A[0, k] = 2
        else: # interior
            A[0, k] = 4

    return A * dx**2 / 4


def correlation_function(corr, mesh, spthresh):
    n = mesh.shape[0]
    c0 = corr['c0']*np.ones((mesh.shape[1], 1))
    c0 = 1/(c0**2)
    sigma = corr['sigma']
    C = np.zeros((n, n))
    for i in range(n):

        # gaussian correlation
        point = mesh[i:i+1, :]
        X = (mesh - point*np.ones((n, 2)))**2
        C[:, i] = sigma*np.exp(-np.sqrt(np.matmul(X,c0)))[:, 0]

    C[C<spthresh] = 0

    return C

def random_field(args):
    xv = np.linspace(0, 1, args.resolution) 
    dx = xv[1]-xv[0]
    corr = {'c0': 0.1,
            'sigma': 1.0}

    X, Y = np.meshgrid(xv, xv, indexing='ij')
    mesh = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

    C = correlation_function(corr, mesh, 1.e-12)
    S, U = scipy.sparse.linalg.eigs(C, k=args.n)
    ev = np.abs(np.real(S))
    inds = np.flip(np.argsort(ev))
    ev = ev[inds]
    U = np.real(U)[:, inds]
    return np.diag(ev), U 

def grad_K(K, i, j, dx, dim):
    # gradient of K w.r.t. dimension at location
    dK = 0
    if dim == 1:
        dK = (K[i+1, j]-K[i-1,j])/(2*dx)
    elif dim == 2:
        dK = (K[i, j+1]-K[i, j-1])/(2*dx)
    return dK

def form_matrix(K, xv, dx):
    nx = len(xv)
    A = np.zeros((nx**2, nx**2))
    f = np.zeros((nx**2, 1))

    j = -1
    i = 0

    dx2 = dx**2
    for k in range(nx**2):
        if (np.mod(k, nx)) == 0:
            j += 1
            i = 0

        A[k, k] = K[i, j]*4/(dx2)

        # corners
        if ((i==0) and (j==0)):
            A[k, k+1] = -2*K[i, j]/(dx2)
            A[k, k+nx] = -2*K[i, j]/(dx2)
        elif ((i==(nx-1)) and (j==(nx-1))):
            A[k, k-1] = -2*K[i, j]/(dx2)
            A[k, k-nx] = -2*K[i, j]/(dx2)
        elif ((i==0) and (j==(nx-1))):
            A[k, k+1] = -2*K[i, j]/(dx2)
            A[k, k-nx] = -2*K[i, j]/(dx2)
        elif (i==(nx-1)) and (j==0):
            A[k, k-1] = -2*K[i, j]/(dx2)
            A[k, k+nx] = -2*K[i, j]/(dx2)

        # bondaries
        elif ((i==0) or (j==0) or (i==(nx-1)) or (j==(nx-1))):
            if i == 0:
                gK = grad_K(K, i, j, dx, 2)/(2*dx)
                A[k, k+1] = -2*K[i, j]/dx2
                A[k, k+nx] = -K[i, j]/dx2 - gK
                A[k, k-nx] = -K[i, j]/dx2 + gK
            elif i == (nx-1):
                gK = grad_K(K, i, j, dx, 2)/(2*dx)
                A[k, k-1] = -2*K[i, j]/dx2
                A[k, k+nx] = -K[i, j]/dx2 - gK
                A[k, k-nx] = -K[i, j]/dx2 + gK
            elif j == 0:
                gK = grad_K(K, i, j, dx, 1)/(2*dx)
                A[k, k+nx] = -2*K[i, j]/dx2
                A[k, k-1] = -K[i, j]/dx2 + gK
                A[k, k+1] = -K[i, j]/dx2 - gK
            elif j == (nx-1):
                gK = grad_K(K, i, j, dx, 1)/(2*dx)
                A[k, k-nx] = -2*K[i, j]/dx2
                A[k, k-1] = -K[i, j]/dx2 + gK
                A[k, k+1] = -K[i, j]/dx2 - gK
        
        # interior
        else:
            gK1 = grad_K(K, i, j, dx, 1)/(2*dx)
            gK2 = grad_K(K, i, j, dx, 2)/(2*dx)
            fac = -K[i, j]/dx2
            A[k, k-1] = fac + gK1
            A[k, k+1] = fac - gK1
            A[k, k+nx] = fac - gK2
            A[k, k-nx] = fac + gK2

        x, y = xv[i], xv[j]

        # source function
        if (np.abs(x-0.0625)<=0.0625) and (np.abs(y-0.0625)<=0.0625):
            f[k, 0] = 10
        elif (np.abs(x-1+0.0625)<=0.0625) and (np.abs(y-1+0.0625)<=0.0625):
            f[k, 0] = -10
                
        i += 1
    return A, f

def compute_u(P, K, nx, xv, dx):
    U1, U2 = np.zeros(P.shape), np.zeros(P.shape)
    for i in range(nx):
        for j in range(nx):
            if ((j==0) or (j==(nx-1))) and (i!=0) and (i!=(nx-1)): # bottom or top boundary (no corners)
                U1[i, j] = -K[i, j]*(P[i+1, j]-P[i-1, j])/(2*dx)
            elif ((i==0) or (i==(nx-1))) and (j!=0) and (j!=(nx-1)): # left or right boundary
                U2[i, j] = -K[i, j]*(P[i, j+1]-P[i, j-1])/(2*dx)
            elif (i==0 and j==0) or (i==(nx-1) and j==(nx-1)) or (i==0 and j==(nx-1)) or (j==0 and i==(nx-1)): # corners
                U1[i, j] = 0
                U2[i, j] = 0
            else: # interior
                U1[i, j] = -K[i, j]*(P[i+1, j]-P[i-1, j])/(2*dx)
                U2[i, j] = -K[i, j]*(P[i, j+1]-P[i, j-1])/(2*dx)

    return U1, U2

def generate_data(i, args, UL, xv, dx, Areg):
    if np.mod(i, 100)==0:
        print('  '+str(i)+' of '+str(args.samples))

    W = np.random.rand(args.n, 1)*2.5
    K = np.matmul(UL, W)
    K = np.exp(K).reshape((args.resolution, args.resolution))
    A, f = form_matrix(K, xv, dx)

    A = np.concatenate((A, Areg), axis=0)
    f = np.concatenate((f, np.zeros((1,1))), axis=0)
    P, _, _, _ = scipy.linalg.lstsq(A, f)
    P = P.reshape((args.resolution, args.resolution))

    U1, U2 = compute_u(P, K, args.resolution, xv, dx)


    # save everything
    np.save(osp.join(args.output, "data", "P_"+str(i)+".npy"), P)
    np.save(osp.join(args.output, "data", "U1_"+str(i)+".npy"), U1)
    np.save(osp.join(args.output, "data", "U2_"+str(i)+".npy"), U2)
    np.save(osp.join(args.output, "permeability", "K_"+str(i)+".npy"), K)
    np.save(osp.join(args.output, "params", "W_"+str(i)+".npy"), W)

    # save image visualizations for only some samples
    if i < 100:
        X, Y = np.meshgrid(xv, xv)
        plt.figure(1)
        plt.clf()
        plt.imshow(P)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig(osp.join(args.output, "imgs", "P_"+str(i)+".png"), bbox_inches='tight', dpi=300)

        plt.figure(2)
        plt.clf()
        plt.imshow(U1)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig(osp.join(args.output, "imgs", "U1_"+str(i)+".png"), bbox_inches='tight', dpi=300)

        plt.figure(3)
        plt.clf()
        plt.imshow(U2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig(osp.join(args.output, "imgs", "U2_"+str(i)+".png"), bbox_inches='tight', dpi=300)

        plt.figure(4)
        plt.clf()
        plt.imshow(K)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar()
        plt.savefig(osp.join(args.output, "imgs", "K_"+str(i)+".png"), bbox_inches='tight', dpi=300)

    return 0


def main():
    args = parse_args()
    if args.resume > 0:
        resume = True
        print('Resuming generation from ', args.resume)
    else:
        resume = False
    create_output_dirs(args, resume)

    print('Computing random field...')
    L, U = random_field(args)
    print('  done computing random field!')

    UL = np.matmul(U, np.sqrt(L))

    xv = np.linspace(0, 1, args.resolution)
    dx = xv[1]-xv[0]
    Areg = enforce_integral(xv, dx)

    print('Computing solutions...')
    
    start_time = time.time()
    _ = Parallel(n_jobs=args.threads)(delayed(generate_data)(i, args, UL, xv, dx, Areg) for i in range(args.resume, args.samples)) 
    '''
    for i in range(8):
        _ = generate_data(i, args, UL, xv, dx, Areg)
    '''

    end_time = time.time()
    print('Total time elapsed: ', end_time-start_time)


if __name__ == "__main__":
    main()

