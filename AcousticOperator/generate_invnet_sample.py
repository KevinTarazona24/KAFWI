import os
import argparse
import torch
import numpy as np


# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================================
# Argumentos
# =========================================================
parser = argparse.ArgumentParser(
    description="Generación de datos sísmicos a partir de un modelo de velocidad"
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="Ruta al archivo .npy que contiene los modelos de velocidad"
)

parser.add_argument(
    "-s",
    "--sample",
    type=int,
    required=True,
    help="Índice del sample que se desea utilizar"
)

args = parser.parse_args()

MODEL_PATH = args.model
SAMPLE_IDX = args.sample


# =========================================================
# Configuración
# =========================================================
SAVE_DATA_DIR = "Data/data_custom"
SAVE_MODEL_DIR = "Data/model_custom"
SAVE_TXT_DIR = "split_files_custom"

SAVE_NAME = f"sample_{SAMPLE_IDX}"

os.makedirs(SAVE_DATA_DIR, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
os.makedirs(SAVE_TXT_DIR, exist_ok=True)


# =========================================================
# Funciones auxiliares
# =========================================================
def denormalize_labels(
    labels,
    original_min_val=1500,
    original_max_val=4500,
    normalized_min_range=0,
    normalized_max_range=1
):
    norm_min = torch.tensor(
        normalized_min_range,
        device=labels.device,
        dtype=labels.dtype
    )

    norm_max = torch.tensor(
        normalized_max_range,
        device=labels.device,
        dtype=labels.dtype
    )

    denormalized_labels = (
        (labels - norm_min)
        / (norm_max - norm_min)
        * (original_max_val - original_min_val)
        + original_min_val
    )

    return denormalized_labels


def ricker(f0, time_values, t0=None, a=None):

    if t0 is None:
        t0 = 1 / f0

    if a is None:
        a = 1

    r = torch.pi * f0 * (time_values - t0)

    wavelet = (
        a
        * (1 - 2 * r ** 2)
        * torch.exp(-r ** 2)
    )

    return wavelet


def padvel(v0, nbc, dx):

    v = torch.cat(
        [
            v0[:, 0:1].repeat(1, nbc),
            v0,
            v0[:, -1:].repeat(1, nbc)
        ],
        dim=1
    )

    v = torch.cat(
        [
            v[0:1, :].repeat(nbc, 1),
            v,
            v[-1:, :].repeat(nbc, 1)
        ],
        dim=0
    )

    return v


def expand_source(s0, nt):

    nt0 = len(s0)

    if nt0 < nt:

        s = torch.zeros(
            nt,
            device=s0.device,
            dtype=s0.dtype
        )

        s[:nt0] = s0

    else:

        s = s0

    return s


def adjust_sr(coord, dx, nbc):

    isx = torch.round(coord["sx"] / dx).int() + 1 + nbc
    isz = torch.round(coord["sz"] / dx).int() + 1 + nbc

    igx = torch.round(coord["gx"] / dx).int() + 1 + nbc
    igz = torch.round(coord["gz"] / dx).int() + 1 + nbc

    if torch.abs(coord["sz"]) < 0.5:
        isz += 1

    igz = igz + (coord["gz"] < 0.5).int()

    return isx, isz, igx, igz


def AbcCoef2D(vel, nbc, dx):

    nzbc, nxbc = vel.shape

    velmin = torch.min(vel)

    nz = nzbc - 2 * nbc
    nx = nxbc - 2 * nbc

    a = (nbc - 1) * dx

    device = vel.device

    kappa = (
        3.0
        * velmin
        * torch.log(
            torch.tensor(
                10000000.0,
                device=device,
                dtype=vel.dtype
            )
        )
        / (2.0 * a)
    )

    damp1d = kappa * (
        (
            torch.arange(
                nbc,
                device=device,
                dtype=vel.dtype
            )
            * dx
            / a
        ) ** 2
    )

    damp = torch.zeros(
        (nzbc, nxbc),
        device=device,
        dtype=vel.dtype
    )

    for iz in range(nzbc):

        damp[iz, :nbc] = damp1d.flip(0)

        damp[
            iz,
            nx + nbc:nx + 2 * nbc
        ] = damp1d[:]

    for ix in range(nbc, nbc + nx):

        damp[:nbc, ix] = damp1d.flip(0)

        damp[
            nbc + nz:nz + 2 * nbc,
            ix
        ] = damp1d[:]

    return damp


def a2d_mod_abc28(
    vel,
    nbc,
    dx,
    nt,
    dt,
    s,
    coord,
    isFS,
    movie=False
):

    v = padvel(
        vel,
        nbc,
        dx
    )

    abc = AbcCoef2D(
        v,
        nbc,
        dx
    )

    alpha = (
        (v * dt / dx) ** 2
    ).to(v.device)

    kappa = (
        abc * dt
    ).to(v.device)

    c1 = -205 / 72
    c2 = 8 / 5
    c3 = -1 / 5
    c4 = 8 / 315
    c5 = -1 / 560

    temp1 = (
        2
        + 2 * c1 * alpha
        - kappa
    ).to(v.device)

    temp2 = (
        1 - kappa
    ).to(v.device)

    beta_dt = (
        v * dt
    ) ** 2

    s = expand_source(
        s,
        nt
    )

    isx, isz, igx, igz = adjust_sr(
        coord,
        dx,
        nbc
    )

    p1 = torch.zeros_like(v)
    p0 = torch.zeros_like(v)

    registros = []

    for it in range(nt):

        p = (
            temp1 * p1
            - temp2 * p0

            + alpha
            * (
                c2
                * (
                    torch.roll(p1, 1, 0)
                    + torch.roll(p1, -1, 0)
                    + torch.roll(p1, 1, 1)
                    + torch.roll(p1, -1, 1)
                )

                + c3
                * (
                    torch.roll(p1, 2, 0)
                    + torch.roll(p1, -2, 0)
                    + torch.roll(p1, 2, 1)
                    + torch.roll(p1, -2, 1)
                )

                + c4
                * (
                    torch.roll(p1, 3, 0)
                    + torch.roll(p1, -3, 0)
                    + torch.roll(p1, 3, 1)
                    + torch.roll(p1, -3, 1)
                )

                + c5
                * (
                    torch.roll(p1, 4, 0)
                    + torch.roll(p1, -4, 0)
                    + torch.roll(p1, 4, 1)
                    + torch.roll(p1, -4, 1)
                )
            )
        )

        source_term = torch.zeros_like(p)

        source_term[
            isz,
            isx
        ] = beta_dt[
            isz,
            isx
        ] * s[it]

        p = p + source_term

        if isFS:

            p[nbc, :] = 0.0

            p[
                nbc - 1:nbc - 4:-1,
                :
            ] = -p[
                nbc + 1:nbc + 5,
                :
            ]

        rec = p[
            igz,
            igx
        ]

        registros.append(rec)

        p0 = p1.clone()
        p1 = p.clone()

    seis = torch.stack(
        registros,
        dim=0
    )

    return seis


def acoustic_operator_raw(
    vel,
    nbc,
    dx,
    nt,
    dt,
    s,
    isFS,
    movie,
    nx,
    nz,
    coord_sx,
    coord_gx
):
    """
    Operador acústico SIN normalización final.

    Devuelve shots crudos con shape:

    [batch, nshots, nt, nx]
    """

    batch = vel.shape[0]
    nshots = len(coord_sx)

    shot_list = []

    for b in range(batch):

        for sx in coord_sx:

            coord = {

                "sz": torch.tensor(
                    5.0,
                    device=vel.device,
                    dtype=vel.dtype
                ),

                "gx": coord_gx,

                "gz": torch.full(
                    (len(coord_gx),),
                    5.0,
                    device=vel.device,
                    dtype=vel.dtype
                ),

                "sx": torch.tensor(
                    [sx],
                    device=vel.device,
                    dtype=vel.dtype
                )
            }

            vmodel = vel[
                b,
                0,
                :,
                :
            ]

            seis = a2d_mod_abc28(
                vel=vmodel,
                nbc=nbc,
                dx=dx,
                nt=nt,
                dt=dt,
                s=s,
                coord=coord,
                isFS=isFS,
                movie=movie
            )

            shot_list.append(seis)

    shots = torch.stack(
        shot_list,
        dim=0
    )

    shots = shots.view(
        batch,
        nshots,
        nt,
        nx
    )

    return shots


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    print("\n=======================================")
    print("Configuración")
    print("=======================================")

    print("Model file:", MODEL_PATH)
    print("Sample index:", SAMPLE_IDX)


    # -----------------------------------------------------
    # Verificar archivo
    # -----------------------------------------------------
    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"No se encontró el archivo: {MODEL_PATH}"
        )


    # -----------------------------------------------------
    # Cargar modelos de velocidad
    # -----------------------------------------------------
    velocity = np.load(
        MODEL_PATH
    )

    print(
        "\nOriginal model file shape:",
        velocity.shape
    )


    # -----------------------------------------------------
    # Verificar índice
    # -----------------------------------------------------
    if SAMPLE_IDX < 0 or SAMPLE_IDX >= velocity.shape[0]:

        raise IndexError(
            f"Sample {SAMPLE_IDX} fuera de rango. "
            f"El archivo contiene {velocity.shape[0]} samples "
            f"(índices 0 a {velocity.shape[0] - 1})."
        )


    # -----------------------------------------------------
    # Seleccionar un solo sample
    # -----------------------------------------------------
    selected_model_np = velocity[
        SAMPLE_IDX:SAMPLE_IDX + 1
    ]

    print(
        "Selected model original shape:",
        selected_model_np.shape
    )


    # -----------------------------------------------------
    # Guardar sample exacto seleccionado
    # -----------------------------------------------------
    selected_model_save_path = os.path.join(
        SAVE_MODEL_DIR,
        f"{SAVE_NAME}_model_selected.npy"
    )

    np.save(
        selected_model_save_path,
        selected_model_np.astype(np.float32)
    )

    print(
        "Saved selected model sample in:",
        selected_model_save_path
    )

    print(
        "Selected model saved shape:",
        selected_model_np.shape
    )


    # -----------------------------------------------------
    # Asegurar shape [1, 1, 70, 70]
    # -----------------------------------------------------
    label_np = selected_model_np

    if label_np.ndim == 3:

        label_np = label_np[
            :,
            None,
            :,
            :
        ]

    print(
        "Label shape after check:",
        label_np.shape
    )


    vel = torch.tensor(
        label_np,
        dtype=torch.float32,
        device=device
    )

    print(
        "Velocity min before check:",
        vel.min().item()
    )

    print(
        "Velocity max before check:",
        vel.max().item()
    )


    # -----------------------------------------------------
    # Si el modelo está normalizado → convertir a m/s
    # -----------------------------------------------------
    if vel.max() <= 10:

        vel_physical = denormalize_labels(
            vel,
            original_min_val=1500,
            original_max_val=4500
        )

        print(
            "Velocity was denormalized to m/s "
            "for forward modeling."
        )

    else:

        vel_physical = vel

        print(
            "Velocity already seems to be in m/s."
        )


    print(
        "Velocity min used in forward:",
        vel_physical.min().item()
    )

    print(
        "Velocity max used in forward:",
        vel_physical.max().item()
    )


    # -----------------------------------------------------
    # Guardar modelo físico
    # -----------------------------------------------------
    model_physical_np = (
        vel_physical
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    model_physical_save_path = os.path.join(
        SAVE_MODEL_DIR,
        f"{SAVE_NAME}_model_physical.npy"
    )

    np.save(
        model_physical_save_path,
        model_physical_np
    )

    print(
        "Saved physical model used for forward modeling in:",
        model_physical_save_path
    )

    print(
        "Physical model saved shape:",
        model_physical_np.shape
    )


    # -----------------------------------------------------
    # Parámetros sísmicos
    # -----------------------------------------------------
    nx = 70
    nz = 70

    dx = 10.0

    dt = 1e-3
    nt = 1000

    freq = 15

    nbc = 120

    isFS = False
    movie = False


    time_values = torch.linspace(
        0,
        1,
        nt,
        device=device
    )

    s = ricker(
        freq,
        time_values
    )


    # -----------------------------------------------------
    # Receptores
    # -----------------------------------------------------
    coord_gx = torch.arange(
        0,
        nx * dx,
        dx,
        device=device,
        dtype=torch.float32
    )


    # -----------------------------------------------------
    # Fuentes / shots
    # -----------------------------------------------------
    sx = 175

    coord_sx = torch.arange(
        0,
        nx * dx + sx,
        sx,
        device=device,
        dtype=torch.float32
    )

    # Garantizar exactamente 5 shots
    coord_sx = coord_sx[:5]


    print(
        "\ncoord_sx:",
        coord_sx
    )

    print(
        "Number of shots:",
        len(coord_sx)
    )

    print(
        "Number of receivers:",
        len(coord_gx)
    )


    # -----------------------------------------------------
    # Generar shots crudos SIN normalización
    # -----------------------------------------------------
    with torch.no_grad():

        data = acoustic_operator_raw(

            vel=vel_physical,

            nbc=nbc,

            dx=dx,

            nt=nt,

            dt=dt,

            s=s,

            isFS=isFS,

            movie=movie,

            nx=nx,

            nz=nz,

            coord_sx=coord_sx,

            coord_gx=coord_gx
        )


    print(
        "\nGenerated raw seismic data shape:",
        data.shape
    )

    print(
        "Raw seismic min:",
        data.min().item()
    )

    print(
        "Raw seismic max:",
        data.max().item()
    )


    # -----------------------------------------------------
    # Guardar data sísmica raw
    # -----------------------------------------------------
    data_np = (
        data
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


    save_data_path = os.path.join(
        SAVE_DATA_DIR,
        f"{SAVE_NAME}_data.npy"
    )


    np.save(
        save_data_path,
        data_np
    )


    print(
        "Saved raw seismic data in:",
        save_data_path
    )

    print(
        "Saved data shape:",
        data_np.shape
    )


    # -----------------------------------------------------
    # Guardar label/modelo para InversionNet
    # -----------------------------------------------------
    save_label_path = os.path.join(
        SAVE_MODEL_DIR,
        f"{SAVE_NAME}_label.npy"
    )


    label_save_np = label_np.astype(
        np.float32
    )


    np.save(
        save_label_path,
        label_save_np
    )


    print(
        "Saved label in:",
        save_label_path
    )

    print(
        "Saved label shape:",
        label_save_np.shape
    )


    # -----------------------------------------------------
    # Crear archivo txt para InversionNet
    # -----------------------------------------------------
    txt_path = os.path.join(
        SAVE_TXT_DIR,
        f"{SAVE_NAME}_val.txt"
    )


    with open(
        txt_path,
        "w"
    ) as f:

        f.write(
            f"{save_data_path} {save_label_path}\n"
        )


    print(
        "Saved annotation txt in:",
        txt_path
    )


    # -----------------------------------------------------
    # Resumen
    # -----------------------------------------------------
    print("\n=======================================")
    print("Done.")
    print("=======================================")

    print(
        "Model file:",
        MODEL_PATH
    )

    print(
        "Selected sample:",
        SAMPLE_IDX
    )

    print(
        "\nGenerated files:"
    )

    print(
        "1) Raw seismic data:",
        save_data_path
    )

    print(
        "2) Label for InversionNet:",
        save_label_path
    )

    print(
        "3) Selected model sample:",
        selected_model_save_path
    )

    print(
        "4) Physical model used in forward:",
        model_physical_save_path
    )

    print(
        "5) Annotation txt:",
        txt_path
    )

    print(
        "\nUse --global-sample-idx 0 "
        "because this file contains only one sample."
    )