import os
import sys
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio
)
from matplotlib.colors import ListedColormap

# =========================================================
# Importaciones del proyecto
# =========================================================
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

from AcousticOperator.acoustic_operator import *
from inversion_network import *
from newnet import FeatureFusionDense

from utils import (
    set_all_seeds,
    AnisotropicTVLoss,
    TOTAL_LOSS,
    rescale_using_meanstd,
    SNR,
    MAE
)


# =========================================================
# Argumentos
# =========================================================
def parse_sample_slice(sample_arg):
    """
    Permite usar:
    - "0"       -> sample 0
    - "209"     -> sample 209
    - "209:210" -> rango tipo Python
    - "[209:210]" también funciona
    """
    sample_arg = str(sample_arg).strip()
    sample_arg = sample_arg.replace("[", "").replace("]", "")

    if ":" in sample_arg:
        start_str, end_str = sample_arg.split(":")
        start = int(start_str)
        end = int(end_str)

        if end <= start:
            raise ValueError(
                f"El rango debe tener end > start. Recibido: {start}:{end}"
            )

        return start, end

    idx = int(sample_arg)
    return idx, idx + 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prediction with transformation block and safe normalized metrics."
    )

    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        type=str,
        help="Ruta del archivo .npy con los modelos de velocidad. Ejemplo: Data/model/velocity.npy"
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        required=True,
        type=str,
        help="Ruta del checkpoint .pt. Ejemplo: Unsupervised_Approach/checkpoints/GeoDWI.pt"
    )

    parser.add_argument(
        "-s",
        "--sample",
        required=True,
        type=str,
        help='Sample o rango. Ejemplos: "0", "209", "209:210", "[209:210]"'
    )

    parser.add_argument(
        "--results-dir",
        default=None,
        type=str,
        help="Carpeta donde se guardarán resultados. Si no se indica, se crea automáticamente."
    )

    parser.add_argument(
        "--label-min",
        default=1500.0,
        type=float,
        help="Velocidad mínima para normalizar/desnormalizar."
    )

    parser.add_argument(
        "--label-max",
        default=4500.0,
        type=float,
        help="Velocidad máxima para normalizar/desnormalizar."
    )

    parser.add_argument("--nx", default=70, type=int)
    parser.add_argument("--nz", default=70, type=int)
    parser.add_argument("--dx", default=10.0, type=float)
    parser.add_argument("--dt", default=1e-3, type=float)
    parser.add_argument("--nt", default=1000, type=int)
    parser.add_argument("--freq", default=15.0, type=float)
    parser.add_argument("--nbc", default=120, type=int)
    parser.add_argument("--sx", default=175.0, type=float)
    parser.add_argument("--nshots", default=5, type=int)

    parser.add_argument(
        "--isFS",
        action="store_true",
        help="Activar condición de superficie libre."
    )

    parser.add_argument(
        "--movie",
        action="store_true",
        help="Flag movie del operador acústico."
    )

    return parser.parse_args()


# =========================================================
# Normalización/desnormalización segura
# =========================================================
def normalize_velocity_physical(v, label_min=1500.0, label_max=4500.0):
    """
    Convierte velocidad física en m/s a rango [0,1].
    """
    return (v - label_min) / (label_max - label_min)


def denormalize_velocity(v_norm, label_min=1500.0, label_max=4500.0):
    """
    Convierte velocidad normalizada [0,1] a m/s.
    """
    return v_norm * (label_max - label_min) + label_min


def prepare_ground_truth_velocity(model_np, label_min=1500.0, label_max=4500.0):
    """
    Recibe un modelo desde .npy y retorna:
    - gt_physical: velocidad en m/s para operador acústico
    - gt_norm: velocidad normalizada en [0,1] para métricas

    Acepta shapes:
    - [N, H, W]
    - [N, 1, H, W]
    """
    if model_np.ndim == 3:
        model_np = model_np[:, None, :, :]
    elif model_np.ndim == 4:
        pass
    else:
        raise ValueError(
            f"Shape inválido para modelo: {model_np.shape}. "
            "Esperado [N,H,W] o [N,1,H,W]."
        )

    gt = torch.from_numpy(model_np).float()

    print("Loaded GT shape:", gt.shape)
    print("Loaded GT min:", gt.min().item())
    print("Loaded GT max:", gt.max().item())

    if gt.max() <= 10.0:
        gt_norm = torch.clamp(gt, 0.0, 1.0)
        gt_physical = denormalize_velocity(
            gt_norm,
            label_min=label_min,
            label_max=label_max
        )
        print("GT appears normalized. Converted to physical velocity for forward modeling.")
    else:
        gt_physical = gt
        gt_norm = normalize_velocity_physical(
            gt_physical,
            label_min=label_min,
            label_max=label_max
        )
        gt_norm = torch.clamp(gt_norm, 0.0, 1.0)
        print("GT appears physical. Converted to normalized [0,1] for metrics.")

    print("GT physical min:", gt_physical.min().item())
    print("GT physical max:", gt_physical.max().item())
    print("GT normalized min:", gt_norm.min().item())
    print("GT normalized max:", gt_norm.max().item())

    return gt_physical, gt_norm


# =========================================================
# Colormap estilo OpenFWI
# =========================================================
rainbow_cmap = ListedColormap(np.load("rainbow256.npy"))


# =========================================================
# Función de ploteo estilo OpenFWI
# =========================================================
def plot_velocity_openfwi_style(output, target, base_path, vmin=None, vmax=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))

    if vmin is None or vmax is None:
        vmax = np.max(target)
        vmin = np.min(target)

    im = ax[0].matshow(output, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title("Prediction", y=1.08)

    ax[1].matshow(target, cmap=rainbow_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title("Ground Truth", y=1.08)

    for axis in ax:
        axis.set_xticks(range(0, 70, 10))
        axis.set_xticklabels(range(0, 700, 100))
        axis.set_yticks(range(0, 70, 10))
        axis.set_yticklabels(range(0, 700, 100))
        axis.set_ylabel("Depth (m)", fontsize=12)
        axis.set_xlabel("Horizontal distance (m)", fontsize=12)

    fig.colorbar(im, ax=ax, shrink=0.75, label="P-wave velocity (m/s)")

    plt.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_path}.svg", bbox_inches="tight", format="svg")
    plt.close(fig)


# =========================================================
# Visualizar canales del bloque transformante
# =========================================================
def plot_reduction_channels(block_estimated, base_path):
    """
    Grafica los canales generados por TransformNet2.
    block_estimated shape esperado: [B, C, H, W]
    """
    block_np = block_estimated[0].detach().cpu().numpy()
    n_channels = block_np.shape[0]

    print("block_estimated shape:", block_estimated.shape)
    print("Number of feature-map channels:", n_channels)

    fig, ax = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))

    if n_channels == 1:
        ax = [ax]

    for c in range(n_channels):
        channel = block_np[c]

        im = ax[c].imshow(
            channel,
            aspect="auto",
            cmap="seismic"
        )

        ax[c].set_title(f"Feature map {c + 1}")
        ax[c].set_xlabel("Width")
        ax[c].set_ylabel("Height")

        plt.colorbar(im, ax=ax[c], fraction=0.046, pad=0.04)

    plt.suptitle("Transformation block feature maps", fontsize=14, fontweight="bold")
    plt.tight_layout()

    plt.savefig(f"{base_path}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{base_path}.svg", bbox_inches="tight", format="svg")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    args = parse_args()

    # =====================================================
    # Reproducibilidad
    # =====================================================
    set_all_seeds(0)

    # =====================================================
    # Device
    # =====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # =====================================================
    # Carpetas
    # =====================================================
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    sample_start, sample_end = parse_sample_slice(args.sample)

    model_base = os.path.splitext(os.path.basename(args.model_path))[0]
    sample_name = (
        f"{model_base}_sample_{sample_start}"
        if sample_end == sample_start + 1
        else f"{model_base}_samples_{sample_start}_{sample_end}"
    )

    if args.results_dir is None:
        RESULTS_DIR = os.path.join(
            BASE_DIR,
            "results",
            f"with_transform_pred_{sample_name}"
        )
    else:
        RESULTS_DIR = args.results_dir

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n================ CONFIGURATION ================")
    print("Model path:", args.model_path)
    print("Checkpoint:", args.checkpoint)
    print("Sample range:", f"{sample_start}:{sample_end}")
    print("Results dir:", RESULTS_DIR)

    # =====================================================
    # Cargar modelo de velocidad
    # =====================================================
    velocity_all = np.load(args.model_path)

    print("\nFull model file shape:", velocity_all.shape)

    if sample_start < 0 or sample_end > velocity_all.shape[0]:
        raise IndexError(
            f"Sample range {sample_start}:{sample_end} fuera de rango. "
            f"El archivo tiene {velocity_all.shape[0]} samples."
        )

    selected_np = velocity_all[sample_start:sample_end].astype(np.float32)

    # =====================================================
    # Preparar GT físico y normalizado
    # =====================================================
    gt_physical, gt_norm = prepare_ground_truth_velocity(
        selected_np,
        label_min=args.label_min,
        label_max=args.label_max
    )

    gt_physical = gt_physical.to(device)
    gt_norm = gt_norm.to(device)

    # =====================================================
    # Parámetros sísmicos
    # =====================================================
    nbc = args.nbc
    dx = args.dx
    sx = args.sx
    nx = args.nx
    nz = args.nz

    coord_gx = torch.arange(
        0,
        nx * dx,
        dx,
        device=device,
        dtype=torch.float32
    )

    coord_sx = torch.arange(
        0,
        nx * dx + sx,
        sx,
        device=device,
        dtype=torch.float32
    )

    coord_sx = coord_sx[:args.nshots]

    nt = args.nt
    dt = args.dt
    freq = args.freq

    s = ricker(freq, torch.linspace(0, 1, steps=nt, device=device))
    isFS = args.isFS
    movie = args.movie

    print("\n================ ACQUISITION ================")
    print("nx:", nx)
    print("nz:", nz)
    print("dx:", dx)
    print("dt:", dt)
    print("nt:", nt)
    print("freq:", freq)
    print("nbc:", nbc)
    print("coord_sx:", coord_sx)
    print("Number of shots:", len(coord_sx))
    print("Number of receivers:", len(coord_gx))

    # =====================================================
    # Shot observado generado desde el modelo físico
    # =====================================================
    with torch.no_grad():
        x = acoustic_operator(
            gt_physical,
            nbc, dx, nt, dt, s, isFS, movie,
            nx, nz, coord_sx, coord_gx
        )

    print("\nObserved seismic data shape:", x.shape)
    print("Observed seismic min:", x.min().item())
    print("Observed seismic max:", x.max().item())

    # =====================================================
    # Preprocesamiento del dato sísmico igual que entrenamiento
    # =====================================================
    x = rescale_using_meanstd(x)

    print("x shape after rescale:", x.shape)
    print("x min after rescale:", x.min().item())
    print("x max after rescale:", x.max().item())

    # =====================================================
    # Modelos
    # =====================================================
    block_transform = TransformNet2().to(device)

    inversion_network = FeatureFusionDense(
        base=32,
        in_ch=2,
        out_ch=1
    ).to(device)

    # =====================================================
    # Cargar checkpoint
    # =====================================================
    if os.path.exists(args.checkpoint):
        print(f"\nCargando checkpoint desde: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)

        if "block_transform_state_dict" not in checkpoint:
            raise KeyError(
                "El checkpoint no contiene 'block_transform_state_dict'. "
                "Este script requiere un checkpoint entrenado CON bloque transformante."
            )

        if "inversion_network_state_dict" not in checkpoint:
            raise KeyError(
                "El checkpoint no contiene 'inversion_network_state_dict'."
            )

        block_transform.load_state_dict(checkpoint["block_transform_state_dict"])
        inversion_network.load_state_dict(checkpoint["inversion_network_state_dict"])

        if "epoch" in checkpoint:
            print(f"Modelo cargado correctamente desde la época {checkpoint['epoch'] + 1}")
        else:
            print("Modelo cargado correctamente.")
    else:
        raise FileNotFoundError(f"No se encontró el checkpoint en: {args.checkpoint}")

    # =====================================================
    # Métricas normalizadas seguras
    # =====================================================
    SSIM_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    rg_l = AnisotropicTVLoss(weight=1e-3, reduction="mean")

    # =====================================================
    # Predicción
    # =====================================================
    block_transform.eval()
    inversion_network.eval()

    with torch.no_grad():
        print("\nRealizando predicción con bloque transformante...")

        block_estimated = block_transform(x)

        print("x shape:", x.shape)
        print("block_estimated shape:", block_estimated.shape)
        print("block_estimated min:", block_estimated.min().item())
        print("block_estimated max:", block_estimated.max().item())

        Vmodel_estimated = inversion_network(block_estimated)

        print("\nVmodel_estimated shape:", Vmodel_estimated.shape)
        print("Pred raw min:", Vmodel_estimated.min().item())
        print("Pred raw max:", Vmodel_estimated.max().item())
        print("GT norm min:", gt_norm.min().item())
        print("GT norm max:", gt_norm.max().item())

        # =================================================
        # MÉTRICAS NORMALIZADAS SEGURAS
        # Se clipea SOLO para métricas y visualización.
        # =================================================
        pred_norm_metric = torch.clamp(Vmodel_estimated, 0.0, 1.0)
        gt_norm_metric = torch.clamp(gt_norm, 0.0, 1.0)

        print("\nPred metric min:", pred_norm_metric.min().item())
        print("Pred metric max:", pred_norm_metric.max().item())
        print("GT metric min:", gt_norm_metric.min().item())
        print("GT metric max:", gt_norm_metric.max().item())

        ssim = SSIM_metric(pred_norm_metric, gt_norm_metric)
        mse = torch.mean((pred_norm_metric - gt_norm_metric) ** 2)
        psnr = PSNR_metric(pred_norm_metric, gt_norm_metric)
        snr = SNR(pred_norm_metric, gt_norm_metric)
        mae = torch.mean(torch.abs(pred_norm_metric - gt_norm_metric))

        # =================================================
        # Errores físicos opcionales en m/s
        # =================================================
        gt_denorm = denormalize_velocity(
            gt_norm_metric,
            label_min=args.label_min,
            label_max=args.label_max
        )

        pred_denorm = denormalize_velocity(
            pred_norm_metric,
            label_min=args.label_min,
            label_max=args.label_max
        )

        mse_denorm = torch.mean((pred_denorm - gt_denorm) ** 2)
        mae_denorm = torch.mean(torch.abs(pred_denorm - gt_denorm))

        # =================================================
        # Forward para consistencia sísmica
        # Usamos pred_denorm físicamente válido.
        # =================================================
        Pm = acoustic_operator(
            pred_denorm,
            nbc, dx, nt, dt, s, isFS, movie,
            nx, nz, coord_sx, coord_gx
        )

        Pm = rescale_using_meanstd(Pm)

        total_loss = TOTAL_LOSS(Pm, x) + rg_l(pred_norm_metric)

        metrics = {
            "safe_normalized_metrics": {
                "ssim": ssim.item(),
                "mse": mse.item(),
                "psnr": psnr.item(),
                "snr": snr.item(),
                "mae": mae.item()
            },
            "physical_error_optional": {
                "mse_m2_s2": mse_denorm.item(),
                "mae_m_per_s": mae_denorm.item()
            },
            "physics_loss": total_loss.item()
        }

        print("\n================ SAFE NORMALIZED METRICS ================")
        print(f"SSIM: {ssim.item():.6f}")
        print(f"MSE : {mse.item():.6f}")
        print(f"PSNR: {psnr.item():.6f}")
        print(f"SNR : {snr.item():.6f}")
        print(f"MAE : {mae.item():.6f}")

        print("\n================ OPTIONAL PHYSICAL ERRORS ===============")
        print(f"MSE [(m/s)^2]: {mse_denorm.item():.6f}")
        print(f"MAE [m/s]    : {mae_denorm.item():.6f}")

        print("\n================ PHYSICS LOSS ===========================")
        print(f"Total loss: {total_loss.item():.6f}")

        print("\n================ METRICS DICT ===========================")
        print(metrics)

        # =================================================
        # Preparar modelos para graficar
        # =================================================
        gt_np = gt_denorm[0, 0].detach().cpu().numpy()
        pred_np = pred_denorm[0, 0].detach().cpu().numpy()

        vmin = np.min(gt_np)
        vmax = np.max(gt_np)

        # =================================================
        # Guardar gráficas
        # =================================================
        plot_velocity_openfwi_style(
            pred_np,
            gt_np,
            os.path.join(RESULTS_DIR, "prediction_vs_groundtruth_openfwi"),
            vmin=vmin,
            vmax=vmax
        )

        plot_reduction_channels(
            block_estimated,
            os.path.join(RESULTS_DIR, "transformation_block_channels")
        )

        # =================================================
        # Guardar modelos
        # =================================================
        np.save(
            os.path.join(RESULTS_DIR, "predicted_model_denormalized.npy"),
            pred_np.astype(np.float32)
        )

        np.save(
            os.path.join(RESULTS_DIR, "ground_truth_model_denormalized.npy"),
            gt_np.astype(np.float32)
        )

        np.save(
            os.path.join(RESULTS_DIR, "predicted_model_normalized_safe.npy"),
            pred_norm_metric[0, 0].detach().cpu().numpy().astype(np.float32)
        )

        np.save(
            os.path.join(RESULTS_DIR, "ground_truth_model_normalized.npy"),
            gt_norm_metric[0, 0].detach().cpu().numpy().astype(np.float32)
        )

        np.save(
            os.path.join(RESULTS_DIR, "predicted_model_normalized_raw.npy"),
            Vmodel_estimated[0, 0].detach().cpu().numpy().astype(np.float32)
        )

print("\nPredicción finalizada.")