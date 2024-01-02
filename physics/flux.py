import torch

def Courant_HD(u, dx, dy, dz, gamma):
    cs = torch.sqrt(gamma * u[4] / u[0])  # sound velocity
    stability_adv_x = dx / (torch.max(cs + torch.abs(u[1])) + 1.e-8)
    stability_adv_y = dy / (torch.max(cs + torch.abs(u[2])) + 1.e-8)
    stability_adv_z = dz / (torch.max(cs + torch.abs(u[3])) + 1.e-8)
    stability_adv = torch.min(torch.tensor([stability_adv_x, stability_adv_y, stability_adv_z], device=u.device, dtype=u.dtype))
    return stability_adv

def Courant_vis_HD(dx, dy, dz, eta, zeta):
    visc = 4. / 3. * eta + zeta  # maximum
    stability_dif_x = 0.5 * dx**2 / (visc + 1.e-8)
    stability_dif_y = 0.5 * dy**2 / (visc + 1.e-8)
    stability_dif_z = 0.5 * dz**2 / (visc + 1.e-8)
    stability_dif = torch.min(torch.tensor([stability_dif_x, stability_dif_y, stability_dif_z], device=dx.device, dtype=dx.dtype))
    return stability_dif

def bc_HD(u, mode):
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2

    if mode == 'periodic':
        # left hand side
        u[:, 0:2, 2:-2, 2:-2] = u[:, Nx-2:Nx, 2:-2, 2:-2]  # x
        u[:, 2:-2, 0:2, 2:-2] = u[:, 2:-2, Ny-2:Ny, 2:-2]  # y
        u[:, 2:-2, 2:-2, 0:2] = u[:, 2:-2, 2:-2, Nz-2:Nz]  # z
        # right hand side
        u[:, Nx:Nx+2, 2:-2, 2:-2] = u[:, 2:4, 2:-2, 2:-2]
        u[:, 2:-2, Ny:Ny+2, 2:-2] = u[:, 2:-2, 2:4, 2:-2]
        u[:, 2:-2, 2:-2, Nz:Nz+2] = u[:, 2:-2, 2:-2, 2:4]

    elif mode == 'trans':
        # left hand side
        u[:, 0, 2:-2, 2:-2] = u[:, 3, 2:-2, 2:-2]  # x
        u[:, 2:-2, 0, 2:-2] = u[:, 2:-2, 3, 2:-2]  # y
        u[:, 2:-2, 2:-2, 0] = u[:, 2:-2, 2:-2, 3]  # z
        u[:, 1, 2:-2, 2:-2] = u[:, 2, 2:-2, 2:-2]  # x
        u[:, 2:-2, 1, 2:-2] = u[:, 2:-2, 2, 2:-2]  # y
        u[:, 2:-2, 2:-2, 1] = u[:, 2:-2, 2:-2, 2]  # z
        # right hand side
        u[:, -2, 2:-2, 2:-2] = u[:, -3, 2:-2, 2:-2]
        u[:, 2:-2, -2, 2:-2] = u[:, 2:-2, -3, 2:-2]
        u[:, 2:-2, 2:-2, -2] = u[:, 2:-2, 2:-2, -3]
        u[:, -1, 2:-2, 2:-2] = u[:, -4, 2:-2, 2:-2]
        u[:, 2:-2, -1, 2:-2] = u[:, 2:-2, -4, 2:-2]
        u[:, 2:-2, 2:-2, -1] = u[:, 2:-2, 2:-2, -4]

    elif mode == 'KHI':
        # left hand side
        u[:, 0:2, 2:-2, 2:-2] = u[:, Nx - 2:Nx, 2:-2, 2:-2]  # x
        u[:, 2:-2, 0, 2:-2] = u[:, 2:-2, 3, 2:-2]  # y
        u[:, 2:-2, 2:-2, 0] = u[:, 2:-2, 2:-2, 3]  # z
        u[:, 2:-2, 1, 2:-2] = u[:, 2:-2, 2, 2:-2]  # y
        u[:, 2:-2, 2:-2, 1] = u[:, 2:-2, 2:-2, 2]  # z
        # right hand side
        u[:, Nx:Nx + 2, 2:-2, 2:-2] = u[:, 2:4, 2:-2, 2:-2]
        u[:, 2:-2, -2, 2:-2] = u[:, 2:-2, -3, 2:-2]
        u[:, 2:-2, 2:-2, -2] = u[:, 2:-2, 2:-2, -3]
        u[:, 2:-2, -1, 2:-2] = u[:, 2:-2, -4, 2:-2]
        u[:, 2:-2, 2:-2, -1] = u[:, 2:-2, 2:-2, -4]

    return u

def bc_HD_vis(u, if_periodic=True):
    _, Nx, Ny, Nz = u.shape
    Nx -= 2
    Ny -= 2
    Nz -= 2

    if if_periodic:
        # Apply periodic boundary conditions
        u[:, 0:2, 0:2, 2:-2] = u[:, Nx - 2:Nx, Ny - 2:Ny, 2:-2]  # xByB
        u[:, 0:2, 2:-2, 0:2] = u[:, Nx - 2:Nx, 2:-2, Nz - 2:Nz]  # xBzB
        u[:, 0:2, Ny:Ny + 2, 2:-2] = u[:, Nx - 2:Nx, 2:4, 2:-2]  # xByT
        u[:, 0:2, 2:-2, Nz:Nz + 2] = u[:, Nx - 2:Nx, 2:-2, 2:4]  # xBzT
        u[:, Nx:Nx + 2, 0:2, 2:-2] = u[:, 2:4, Ny - 2:Ny, 2:-2]  # xTyB
        u[:, Nx:Nx + 2, 2:-2, 0:2] = u[:, 2:4, 2:-2, Nz - 2:Nz]  # xTzB
        u[:, Nx:Nx + 2, Ny:Ny + 2, 2:-2] = u[:, 2:4, 2:4, 2:-2]  # xTyT
        u[:, Nx:Nx + 2, 2:-2, Nz:Nz + 2] = u[:, 2:4, 2:-2, 2:4]  # xTzT
    else:
        # Apply trans boundary conditions
        # Note: The original slicing seems incorrect (e.g., 4:2 does not make sense). Adjust as needed.
        u[:, 0:2, 0:2, 2:-2] = u[:, 4:2, 4:2, 2:-2]  # xByT
        u[:, 0:2, 2:-2, 0:2] = u[:, 4:2, 2:-2, 4:2]  # xBzB
        u[:, 0:2, Ny:Ny + 2, 2:-2] = u[:, 4:2, Ny:Ny-2, 2:-2]  # xByB
        u[:, 0:2, 2:-2, Nz:Nz + 2] = u[:, 4:2, 2:-2, Nz:Nz-2]  # xBzT
        u[:, Nx:Nx + 2, 0:2, 2:-2] = u[:, Nx:Nx-2, 4:2, 2:-2]  # xTyB
        u[:, Nx:Nx + 2, 2:-2, 0:2] = u[:, Nx:Nx-2, 2:-2, 4:2]  # xTzB
        u[:, Nx:Nx + 2, Ny:Ny + 2, 2:-2] = u[:, Nx:Nx-2, Ny:Ny-2, 2:-2]  # xTyT
        u[:, Nx:Nx + 2, 2:-2, Nz:Nz + 2] = u[:, Nx:Nx-2, 2:-2, Nz:Nz-2]  # xTzT

    return u

def VLlimiter(a, b, c, alpha=2.):
    return torch.sign(c) * (0.5 + 0.5 * torch.sign(a * b)) * torch.minimum(alpha * torch.minimum(torch.abs(a), torch.abs(b)), torch.abs(c))

import torch

def limiting_HD(u, if_second_order):
    nd, nx, ny, nz = u.shape
    uL, uR = u.clone(), u.clone()  # clone to avoid modifying the original tensor
    nx -= 4

    duL = u[:, 1:nx + 3, :, :] - u[:, 0:nx + 2, :, :]
    duR = u[:, 2:nx + 4, :, :] - u[:, 1:nx + 3, :, :]
    duM = (u[:, 2:nx + 4, :, :] - u[:, 0:nx + 2, :, :]) * 0.5
    gradu = VLlimiter(duL, duR, duM) * if_second_order
    # -1:Ncell
    uL[:, 1:nx + 3, :, :] = u[:, 1:nx + 3, :, :] - 0.5 * gradu  # left of cell
    uR[:, 1:nx + 3, :, :] = u[:, 1:nx + 3, :, :] + 0.5 * gradu  # right of cell

    uL = torch.where(uL[0] > 0., uL, u)
    uL = torch.where(uL[4] > 0., uL, u)
    uR = torch.where(uR[0] > 0., uR, u)
    uR = torch.where(uR[4] > 0., uR, u)

    return uL, uR

def HLLC(QL, QR, direc, gamma, gamminv1, gamgamm1inv):
    """ full-Godunov method -- exact shock solution"""

    iX, iY, iZ = direc + 1, (direc + 1) % 3 + 1, (direc + 2) % 3 + 1
    cfL = torch.sqrt(gamma * QL[4] / QL[0])
    cfR = torch.sqrt(gamma * QR[4] / QR[0])
    Sfl = torch.minimum(QL[iX, 2:-1], QR[iX, 1:-2]) - torch.maximum(cfL[2:-1], cfR[1:-2])  # left-going wave
    Sfr = torch.maximum(QL[iX, 2:-1], QR[iX, 1:-2]) + torch.maximum(cfL[2:-1], cfR[1:-2])  # right-going wave

    UL = torch.zeros_like(QL)
    UR = torch.zeros_like(QR)
    UL[0] = QL[0]
    UL[iX] = QL[0] * QL[iX]
    UL[iY] = QL[0] * QL[iY]
    UL[iZ] = QL[0] * QL[iZ]
    UL[4] = gamminv1 * QL[4] + 0.5 * (UL[iX] * QL[iX] + UL[iY] * QL[iY] + UL[iZ] * QL[iZ])
    UR[0] = QR[0]
    UR[iX] = QR[0] * QR[iX]
    UR[iY] = QR[0] * QR[iY]
    UR[iZ] = QR[0] * QR[iZ]
    UR[4] = gamminv1 * QR[4] + 0.5 * (UR[iX] * QR[iX] + UR[iY] * QR[iY] + UR[iZ] * QR[iZ])

    Va = (Sfr - QL[iX, 2:-1]) * UL[iX, 2:-1] - (Sfl - QR[iX, 1:-2]) * UR[iX, 1:-2] - QL[4, 2:-1] + QR[4, 1:-2]
    Va /= (Sfr - QL[iX, 2:-1]) * QL[0, 2:-1] - (Sfl - QR[iX, 1:-2]) * QR[0, 1:-2]
    Pa = QR[4, 1:-2] + QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) * (Va - QR[iX, 1:-2])

    # shock jump condition
    Dal = QR[0, 1:-2] * (Sfl - QR[iX, 1:-2]) / (Sfl - Va)  # right-hand density
    Dar = QL[0, 2:-1] * (Sfr - QL[iX, 2:-1]) / (Sfr - Va)  # left-hand density

    fL = torch.zeros_like(QL)
    fR = torch.zeros_like(QR)
    fL[0] = UL[iX]
    fL[iX] = UL[iX] * QL[iX] + QL[4]
    fL[iY] = UL[iX] * QL[iY]
    fL[iZ] = UL[iX] * QL[iZ]
    fL[4] = (UL[4] + QL[4]) * QL[iX]
    fR[0] = UR[iX]
    fR[iX] = UR[iX] * QR[iX] + QR[4]
    fR[iY] = UR[iX] * QR[iY]
    fR[iZ] = UR[iX] * QR[iZ]
    fR[4] = (UR[4] + QR[4]) * QR[iX]

    # upwind advection scheme
    far = torch.zeros_like(QL[:, 2:-1])
    fal = torch.zeros_like(QR[:, 1:-2])
    far[0] = Dar * Va
    far[iX] = Dar * Va**2 + Pa
    far[iY] = Dar * Va * QL[iY, 2:-1]
    far[iZ] = Dar * Va * QL[iZ, 2:-1]
    far[4] = (gamgamm1inv * Pa + 0.5 * Dar * (Va**2 + QL[iY, 2:-1]**2 + QL[iZ, 2:-1]**2)) * Va
    fal[0] = Dal * Va
    fal[iX] = Dal * Va**2 + Pa
    fal[iY] = Dal * Va * QR[iY, 1:-2]
    fal[iZ] = Dal * Va * QR[iZ, 1:-2]
    fal[4] = (gamgamm1inv * Pa + 0.5 * Dal * (Va**2 + QR[iY, 1:-2]**2 + QR[iZ, 1:-2]**2)) * Va

    f_Riemann = torch.where(Sfl > 0., fR[:, 1:-2], fL[:, 2:-1])  # Sf2 > 0 : supersonic
    f_Riemann = torch.where(Sfl * Va < 0., fal, f_Riemann)  # SL < 0 and Va > 0 : sub-sonic
    f_Riemann = torch.where(Sfr * Va < 0., far, f_Riemann)  # Va < 0 and SR > 0 : sub-sonic

    return f_Riemann

def flux_x(Q, if_second_order=1., **gamma_kwargs):
    QL, QR = limiting_HD(Q, if_second_order=if_second_order)
    f_Riemann = HLLC(QL, QR, direc=0, **gamma_kwargs)
    return f_Riemann

def flux_y(Q, if_second_order=1., **gamma_kwargs):
    _Q = Q.permute(0, 2, 3, 1)  # (y, z, x)
    QL, QR = limiting_HD(_Q, if_second_order=if_second_order)
    f_Riemann = torch.permute(HLLC(QL, QR, direc=1, **gamma_kwargs), (0, 3, 1, 2))  # (x,y,z) = (Z,X,Y)
    return f_Riemann

def flux_z(Q, if_second_order=1., **gamma_kwargs):
    _Q = Q.permute(0, 3, 1, 2)  # (z, x, y)
    QL, QR = limiting_HD(_Q, if_second_order=if_second_order)
    f_Riemann = torch.permute(HLLC(QL, QR, direc=2, **gamma_kwargs), (0, 2, 3, 1))
    return f_Riemann

def get_flux(snapshot, Nx, Ny, Nz, pad_width=2, mode='periodic', **gamma_kwargs):
    # snapshot: (D, P, ux, uy)
    # Q: (D, ux, uy, uz, P)
    '''
    # torch inplace error for vmap
    Q = torch.zeros((5, Nx+2*pad_width, Ny+2*pad_width, Nz+2*pad_width))
    Q[0, 2:-2, 2:-2, 2:-2] = snapshot[0]
    Q[1, 2:-2, 2:-2, 2:-2] = snapshot[2]
    Q[2, 2:-2, 2:-2, 2:-2] = snapshot[3]
    Q[4, 2:-2, 2:-2, 2:-2] = snapshot[1]
    '''
    padding = (pad_width, pad_width, pad_width, pad_width, pad_width, pad_width)
    # Initialize Q with zeros and pad the entire tensor
    Q = torch.cat((snapshot[0:1], snapshot[2:3], snapshot[3:4], torch.zeros_like(snapshot[0:1]), snapshot[1:2]), dim=0)
    Q = torch.nn.functional.pad(Q, padding, mode='constant')
    Q = bc_HD(Q, mode=mode)
    fx = flux_x(Q, if_second_order=1, **gamma_kwargs)
    fy = flux_y(Q, if_second_order=1, **gamma_kwargs)
    fz = flux_z(Q, if_second_order=1, **gamma_kwargs)
    density_flux =  (fx[0, 1:, 2:-2, 2:-2] - fx[0, :-1, 2:-2, 2:-2])\
            + (fy[0, 2:-2, 1:, 2:-2] - fy[0, 2:-2, :-1, 2:-2])\
            + (fz[0, 2:-2, 2:-2, 1:] - fz[0, 2:-2, 2:-2, :-1])
    Mx =  (fx[1, 1:, 2:-2, 2:-2] - fx[1, :-1, 2:-2, 2:-2])\
                + (fy[1, 2:-2, 1:, 2:-2] - fy[1, 2:-2, :-1, 2:-2])\
                +  (fz[1, 2:-2, 2:-2, 1:] - fz[1, 2:-2, 2:-2, :-1])
    My = (fx[2, 1:, 2:-2, 2:-2] - fx[2, :-1, 2:-2, 2:-2])\
                + (fy[2, 2:-2, 1:, 2:-2] - fy[2, 2:-2, :-1, 2:-2])\
                + (fz[2, 2:-2, 2:-2, 1:] - fz[2, 2:-2, 2:-2, :-1])
    #Mz =(fx[3, 1:, 2:-2, 2:-2] - fx[3, :-1, 2:-2, 2:-2])\
    #            + (fy[3, 2:-2, 1:, 2:-2] - fy[3, 2:-2, :-1, 2:-2])\
    #            + (fz[3, 2:-2, 2:-2, 1:] - fz[3, 2:-2, 2:-2, :-1])
    energy_flux = (fx[4, 1:, 2:-2, 2:-2] - fx[4, :-1, 2:-2, 2:-2])\
                + (fy[4, 2:-2, 1:, 2:-2] - fy[4, 2:-2, :-1, 2:-2])\
                + (fz[4, 2:-2, 2:-2, 1:] - fz[4, 2:-2, 2:-2, :-1])
    
    return density_flux, Mx, My, energy_flux

class FluxCalculator:
    def __init__(self,Nx, Ny, Nz, L=1, pad_width=2, mode='periodic', gamma = 1.4):
        self.L = L
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.dx, self.dy, self.dz = L / Nx, L / Ny, L / Nz
        self.num_cells = Nx * Ny * Nz
        self.pad_width = pad_width
        self.mode = mode
        gamminv1 = 1. / (gamma - 1.)
        gamgamm1inv = gamma * gamminv1
        gamma_kwargs = {'gamma': gamma, 'gamminv1': gamminv1, 'gamgamm1inv': gamgamm1inv}
        self.gamma_kwargs = gamma_kwargs

    def __call__(self, snapshot):
        # snapshot: (D, P, ux, uy)
        # Q: (D, ux, uy, uz, P)
        '''
        # torch inplace error for vmap
        Q = torch.zeros((5, Nx+2*pad_width, Ny+2*pad_width, Nz+2*pad_width))
        Q[0, 2:-2, 2:-2, 2:-2] = snapshot[0]
        Q[1, 2:-2, 2:-2, 2:-2] = snapshot[2]
        Q[2, 2:-2, 2:-2, 2:-2] = snapshot[3]
        Q[4, 2:-2, 2:-2, 2:-2] = snapshot[1]
        '''
        padding = (self.pad_width, self.pad_width, self.pad_width, self.pad_width, self.pad_width, self.pad_width)
        Q = torch.cat((snapshot[0:1], snapshot[2:3], snapshot[3:4], torch.zeros_like(snapshot[0:1]), snapshot[1:2]), dim=0)
        Q = torch.nn.functional.pad(Q, padding, mode='constant')
        Q = bc_HD(Q, mode=self.mode)

        fx = flux_x(Q, if_second_order=1, **self.gamma_kwargs)
        fy = flux_y(Q, if_second_order=1, **self.gamma_kwargs)
        fz = flux_z(Q, if_second_order=1, **self.gamma_kwargs)

        # Compute the residuals
        density_flux = self._compute_flux(fx[0], fy[0], fz[0])
        Mx = self._compute_flux(fx[1], fy[1], fz[1])
        My = self._compute_flux(fx[2], fy[2], fz[2])
        energy_flux = self._compute_flux(fx[4], fy[4], fz[4])

        return density_flux, Mx, My, energy_flux

    def _compute_flux(self, fx, fy, fz):
        # Compute residual for a single component
        return (fx[1:, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width] - fx[:-1, self.pad_width:-self.pad_width, self.pad_width:-self.pad_width])/self.dx \
               + (fy[self.pad_width:-self.pad_width, 1:, self.pad_width:-self.pad_width] - fy[self.pad_width:-self.pad_width, :-1, self.pad_width:-self.pad_width])/self.dy \
               + (fz[self.pad_width:-self.pad_width, self.pad_width:-self.pad_width, 1:] - fz[self.pad_width:-self.pad_width, self.pad_width:-self.pad_width, :-1])/self.dz
