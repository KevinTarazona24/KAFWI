# KAFWI
### Physics-Guided Unsupervised Deep Learning for P-Wave Velocity Estimation

KAFWI is a research repository for estimating two-dimensional subsurface P-wave velocity models from reflection seismic data using physics-guided unsupervised deep learning.

The main contribution of this work is the design of a two-branch neural architecture that combines complementary representations of seismic observations to estimate a velocity model. Training is guided by a differentiable acoustic wave-propagation operator, which connects the predicted subsurface model to its simulated seismic response.

## Research Motivation

Full-waveform inversion (FWI) estimates subsurface properties by minimizing the discrepancy between observed and simulated seismic data. Although it can produce high-resolution velocity models, its performance depends on factors such as the initial model, acquisition geometry, frequency content, and computational resources.

Supervised deep-learning approaches offer an alternative, but typically require large collections of paired seismic data and reference velocity models. Such labeled datasets are difficult to obtain for real subsurface conditions.

This project investigates an unsupervised framework in which seismic observations provide the training signal through a physical forward model. Reference velocity models are therefore not required to define the training objective, although they can be used to evaluate predictions in synthetic experiments.

## Method Overview

The framework connects neural velocity estimation with acoustic wave simulation:

1. **Seismic input:** reflection seismic records are provided to the neural network.
2. **Feature extraction and fusion:** a two-branch architecture processes complementary information from the input data.
3. **Velocity prediction:** the network produces a two-dimensional P-wave velocity model.
4. **Forward simulation:** a differentiable acoustic operator generates synthetic seismic data from the predicted model.
5. **Loss evaluation:** the simulated response is compared with the observed seismic data, together with spatial regularization.
6. **Parameter optimization:** gradients are propagated through the forward operator to update the neural network.

The differentiable operator enables the calculation of gradients linking the seismic-data discrepancy to the predicted velocity model and, subsequently, to the network parameters.

## Optimization Objective

The training objective combines:

- **L1 seismic misfit**, measuring absolute differences between observed and simulated seismic data.
- **L2 seismic misfit**, penalizing squared differences between the seismic responses.
- **Anisotropic total variation**, encouraging spatial regularity in the estimated velocity model.

A depth-dependent gradient weighting is also considered to adjust the contribution of different depth regions during optimization.

The balance between data fitting and regularization is important: the estimated model should explain the seismic observations while maintaining a geologically interpretable spatial structure.

## Repository Structure

```text
KAFWI/
├── AcousticOperator/         # Acoustic forward-modeling components
├── Unsupervised_Approach/    # Neural-network and inversion components
├── main.py                  # Main experiment script
├── utils.py                 # Supporting utilities
├── rainbow256.npy           # Auxiliary NumPy resource
├── README.md                # Project documentation
└── .gitignore               # Git exclusion rules
```

## Experimental Scope

The research focuses on synthetic reflection seismic experiments, where reference velocity models are available for assessing reconstruction quality.

Evaluation considers the recovery of subsurface velocity structures across different geological scenarios, including layered media and folded structures such as antiforms and synforms.

The analysis examines both the agreement between observed and simulated seismic responses and the correspondence between estimated and reference velocity models. These assessments provide complementary information: a low seismic misfit alone does not guarantee a unique or geologically accurate reconstruction.

## Reproducibility

To obtain a local copy of the repository:

```bash
git clone https://github.com/KevinTarazona24/KAFWI.git
cd KAFWI
```

Before running an experiment, review the source code and configure the input data, acquisition parameters, computational device, and optimization settings for the intended case.

Experiment reproduction requires consistent spatial and temporal sampling, source definitions, receiver locations, and forward-modeling settings.

## Limitations

This implementation is intended for research and experimentation. Reconstruction quality depends on the information contained in the seismic observations, the assumptions of the acoustic model, the network architecture, and the optimization configuration.

Results obtained on synthetic datasets should not be interpreted as evidence of equivalent performance on field data without further validation.

## Academic Context

This repository supports research on physics-guided unsupervised deep learning for P-wave velocity estimation from reflection seismic data.

The architectural contribution is developed within an existing physics-guided learning paradigm: the physical operator supplies the connection between the estimated model and the seismic observations, while the proposed neural architecture defines how the input data are transformed into a velocity estimate.
