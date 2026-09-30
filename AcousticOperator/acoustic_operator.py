import torch
import time
import matplotlib.pyplot as plt
import numpy as np

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_data(x):
    # Calcular el valor mínimo del tensor. Esto es diferenciable.
    min_val = x.min()
    
    # Calcular el valor máximo del tensor. Esto es diferenciable.
    max_val = x.max()
    
    # Si todos los valores en el tensor son iguales, evitamos la división por cero.
    # En este caso, el tensor normalizado será un tensor de ceros.
    # Esta condición es importante para la estabilidad numérica y no rompe el grafo.
    if (max_val - min_val) == 0:
        return torch.zeros_like(x)
    
    # Aplicar la fórmula de normalización min-max: (x - min_x) / (max_x - min_x)
    # Todas estas operaciones aritméticas (+, -, /) son diferenciables en PyTorch.
    normalized_tensor = (x - min_val) / (max_val - min_val)
    
    return normalized_tensor
    
def denormalize_labels(labels, original_min_val=1500, original_max_val=4500, normalized_min_range=0, normalized_max_range=1):

    norm_min = torch.tensor(normalized_min_range, device=labels.device, dtype=labels.dtype)
    norm_max = torch.tensor(normalized_max_range, device=labels.device, dtype=labels.dtype)
    
    denormalized_labels = (labels - norm_min) / (norm_max - norm_min) * (original_max_val - original_min_val) + original_min_val
    
    return denormalized_labels
################################################################################################################
def ricker(f0, time_values, t0=None, a=None):
    if t0 is None:
        t0 = 1 / f0
    if a is None:
        a = 1

    r = torch.pi * f0 * (time_values - t0)
    wavelet = a * (1 - 2 * r ** 2) * torch.exp(-r ** 2)

    return wavelet

################################################################################################################
def padvel(v0, nbc, dx):
    """
    Rellena la matriz v0 con bordes replicados.

    Parameters:
        v0 (numpy.ndarray): Matriz original.
        nbc (int): Número de capas de borde a agregar.

    Returns:
        numpy.ndarray: Matriz con bordes rellenados.
    """

    # Replicar columnas en los bordes
    v = torch.cat([
        v0[:, 0:1].repeat(1, nbc),  # Replicar primera columna
        v0,
        v0[:, -1:].repeat(1, nbc)  # Replicar última columna
    ], dim=1)

    # Replicar filas en los bordes
    v = torch.cat([
        v[0:1, :].repeat(nbc, 1),  # Replicar primera fila
        v,
        v[-1:, :].repeat(nbc, 1)  # Replicar última fila
    ], dim=0)

    return v

################################################################################################################

def expand_source(s0, nt):
    """
    Expande el vector s0 para que tenga tamaño nt.

    Parameters:
        s0 (numpy.ndarray): Vector original.
        nt (int): Tamaño deseado.

    Returns:
        numpy.ndarray: Vector expandido.
    """
    nt0 = len(s0)
    if nt0 < nt:
        s = torch.zeros(nt, device=s0.device)
        s[:nt0] = s0
    else:
        s = s0

    return s

################################################################################################################


def adjust_sr(coord, dx, nbc):

    isx = torch.round(coord['sx'] / dx).int() + 1 + nbc
    isz = torch.round(coord['sz'] / dx).int() + 1 + nbc
    igx = torch.round(coord['gx'] / dx).int() + 1 + nbc
    igz = torch.round(coord['gz'] / dx).int() + 1 + nbc

    if torch.abs(coord['sz']) < 0.5:
        isz += 1

    igz = igz + (coord['gz'] < 0.5).int()

    return isx, isz, igx, igz

################################################################################################################

def AbcCoef2D(vel, nbc, dx):
    nzbc, nxbc = vel.shape
    velmin = torch.min(vel)
    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc
    a = (nbc - 1) * dx
    device = vel.device
    kappa = 3.0 * velmin * torch.log(torch.tensor(10000000.0, device=device)) / (2.0 * a)

    # setup 1D BC damping arrays
    damp1d = kappa * ((torch.arange(nbc, device=device) * dx / a) ** 2)

    # setup 2D BC damping array
    damp = torch.zeros((nzbc, nxbc))

    # fulltill zone 1, 4, 7 and 3, 6, 9
    for iz in range(nzbc):
        # damp[iz, :nbc] = damp1d[::-1]
        damp[iz, :nbc] = damp1d.flip(0)
        damp[iz, nx + nbc:nx + 2 * nbc] = damp1d[:]

    # full fill zone 2 and 8
    for ix in range(nbc, nbc + nx):
        # damp[:nbc, ix] = damp1d[::-1]
        damp[:nbc, ix] = damp1d.flip(0)
        damp[nbc + nz:nz + 2 * nbc, ix] = damp1d[:]

    return damp

################################################################################################################

def a2d_mod_abc28(vel, nbc, dx, nt, dt, s, coord, isFS, movie=False):
    # número de receptores
    ng = len(coord['gx'])

    # padding y coeficientes (igual que antes)
    v = padvel(vel, nbc, dx)
    abc = AbcCoef2D(v, nbc, dx)
    alpha = ((v * dt / dx) ** 2).to(v.device)
    kappa = (abc * dt).to(v.device)
    c1, c2, c3, c4, c5 = -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560
    temp1 = (2 + 2 * c1 * alpha - kappa).to(v.device)
    temp2 = (1 - kappa).to(v.device)
    beta_dt = (v * dt) ** 2

    # wavelet y posiciones (igual que antes)
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)

    # estados anteriores
    p1 = torch.zeros_like(v)
    p0 = torch.zeros_like(v)

    # lista donde guardamos cada registro de nt×ng
    registros = []

    for it in range(nt):
        # esquema FD (igual que antes)
        p = (temp1 * p1
             - temp2 * p0
             + alpha * (
                     c2 * (torch.roll(p1, 1, 0) + torch.roll(p1, -1, 0)
                           + torch.roll(p1, 1, 1) + torch.roll(p1, -1, 1))
                     + c3 * (torch.roll(p1, 2, 0) + torch.roll(p1, -2, 0)
                             + torch.roll(p1, 2, 1) + torch.roll(p1, -2, 1))
                     + c4 * (torch.roll(p1, 3, 0) + torch.roll(p1, -3, 0)
                             + torch.roll(p1, 3, 1) + torch.roll(p1, -3, 1))
                     + c5 * (torch.roll(p1, 4, 0) + torch.roll(p1, -4, 0)
                             + torch.roll(p1, 4, 1) + torch.roll(p1, -4, 1))
             )
             )
        # fuente
        source_term = torch.zeros_like(p)
        source_term[isz, isx] = beta_dt[isz, isx] * s[it]
        p = p + source_term

        #p[isz, isx] = p[isz, isx] + beta_dt[isz, isx] * s[it]
        # free surface (igual)
        if isFS:
            p[nbc, :] = 0.0
            p[nbc - 1:nbc - 4:-1, :] = -p[nbc + 1:nbc + 5, :]

        # en vez del bucle interno y asignaciones, hacemos:
        # p[igz, igx] es un vector de longitud ng
        rec = p[igz, igx]  # shape → (ng,)
        registros.append(rec)  # grabamos ese vector

        # actualizamos estado
        #p0, p1 = p1, p
        p0 = p1.clone()
        p1 = p.clone()

    # apilamos todos los nt registros → tensor (nt, ng)
    seis = torch.stack(registros, dim=0)

    return seis

################################################################################################################

def acoustic_operator(vel, nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx):
    batch = vel.shape[0]
    nshots = len(coord_sx)
    shot_list = []
    #print("grad vel ANTES", vel.requires_grad)  # Debe ser True

    # recolectamos cada seismograma en una lista (no pre-alocación)
    for v in range(batch):
        for sx in coord_sx:
            coord = {
                'sz': torch.tensor(5.0, device=vel.device),
                'gx': coord_gx,
                'gz': torch.full((len(coord_gx),), 5.0, device=vel.device),
                'sx': torch.tensor([sx], device=vel.device)
            }
            vmodel = vel[v,0,:,:]
            #print("grad vel DURANTE", vmodel.requires_grad)  # Debe ser True
            #print('model shape: ', vmodel.shape)
            seis = a2d_mod_abc28(vmodel, nbc, dx, nt, dt, s, coord, isFS, movie)
            shot_list.append(seis)

    # apilamos la lista en un tensor de forma (batch, nshots, nt, nx)
    shots = torch.stack(shot_list, dim=0)
    shots = shots.view(batch, nshots, nt, nx)

    # normalización z-score (manteniendo el grafo)
    #print('ANTES', shots.min())
    #print('ANTES',shots.max())
    shots = normalize_data(shots)
    #print("shots.requires_grad:", shots.requires_grad)  # Debe ser True
	
    return shots

