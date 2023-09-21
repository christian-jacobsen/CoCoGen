import numpy as np
import os
import os.path as osp
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compute mean and stdev of 2D darcy flow data")
    parser.add_argument('--path', type=str, help="Path to dataset root to compute mean and variance of samples and parameters")
    return parser.parse_args()

def main():
    args = parse_args()
    
    sample_names = os.listdir(osp.join(args.path, "data"))
    param_names = os.listdir(osp.join(args.path, "params"))

    n = len(sample_names)
    m = len(param_names)
    print(str(m), " samples available!")

    sample_mean = np.array([0., 0., 0.])
    sample_std = np.array([0., 0., 0.])
    param_mean = 0.
    param_std = 0.

    # compute data means
    for i in range(n):
        x = np.load(osp.join(args.path, "data", sample_names[i]))

        if i == 0:
            print('data shape: ', x.shape)

        # P average
        if sample_names[i][0] == 'P':
            sample_mean[0] = sample_mean[0] + np.sum(x) / (x.shape[0]*x.shape[1])
        # U1 average
        elif sample_names[i][0:2] == 'U1':
            sample_mean[1] = sample_mean[1] + np.sum(x) / (x.shape[0]*x.shape[1])
        # U2 average
        elif sample_names[i][0:2] == 'U2':
            sample_mean[2] = sample_mean[2] + np.sum(x) / (x.shape[0]*x.shape[1])

    # param mean
    for i in range(m):
        p = np.load(osp.join(args.path, "params", param_names[i]))

        if i == 0:
            print('parameter shape: ', p.shape)

        param_mean = param_mean + (np.sum(p) / len(p))

    sample_mean = sample_mean / m
    param_mean = param_mean / m

    # data std
    for i in range(n):
        x = np.load(osp.join(args.path, "data", sample_names[i]))

        # P average
        if sample_names[i][0] == 'P':
            sample_std[0] = sample_std[0] + np.sum((x-sample_mean[0])**2) / (x.shape[0]*x.shape[1])
        # U1 average
        elif sample_names[i][0:2] == 'U1':
            sample_std[1] = sample_std[1] + np.sum((x-sample_mean[1])**2) / (x.shape[0]*x.shape[1])
        # U2 average
        elif sample_names[i][0:2] == 'U2':
            sample_std[2] = sample_std[2] + np.sum((x-sample_mean[2])**2) / (x.shape[0]*x.shape[1])

    # param std
    for i in range(m):
        p = np.load(osp.join(args.path, "params", param_names[i]))
        param_std += np.sum((p-param_mean)**2) / len(p)

    sample_std = np.sqrt(sample_std / m)
    param_std = np.sqrt(param_std / m)

    print("Data mean: [P, U1, U2] = "+str(sample_mean))
    print("Data std: [P, U1, U2] = "+str(sample_std))
    print("Param mean: "+str(param_mean))
    print("Param std: "+str(param_std))


if __name__ == "__main__":
    main()
