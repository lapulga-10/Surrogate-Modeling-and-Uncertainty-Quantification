# Surrogate Modeling & UQ: Solar Sail Membranes

## Project Overview
This project develops surrogate models (SVR, Polynomial, GPR) to approximate computationally expensive finite element simulations of maximum Cauchy stress in prestressed solar sail membranes. Optimized Latin Hypercube sampling generates a 200-point dataset from Kratos Multiphysics. GPR excels with 90%+ MSE reduction vs SVR, enabling Monte Carlo uncertainty quantification revealing right-skewed stress distribution (mean 5312 kPa).

## Dataset
Synthetic dataset from **200 high-fidelity FEM simulations** (noise-free):
- **Inputs** (6 random vars): Membrane Young's modulus (lognormal), Poisson's ratio (uniform), pre-stress (lognormal), surface loading (Gumbel), edge/support cable pre-stresses (lognormal).
- **Output**: Maximal membrane Cauchy stress (kPa).
- Split: 80% train (160), 20% test (40).

## Objectives
1. Compare surrogate model accuracy via residuals & MSE.
2. Identify best surrogate (GPR) for uncertainty propagation.
3. Characterize output distribution via Monte Carlo (10k samples) & parametric fits (lognormal/gamma).

## Statistical Methods
- **Optimized LHS**: Genetic algorithm space-filling design in 6D input space.
- **Surrogates**:
  - SVR (RBF kernel, ε-insensitive loss)
  - Polynomial regression (multivariate degree d)
  - **GPR** (RBF kernel, noise-free posterior)
- **UQ**: Monte Carlo w/ GPR + AIC for lognormal/gamma fits.

## Key Findings
1. **Model Performance**:
   | Model       | Train MSE | Test MSE | Improvement vs SVR |
   |-------------|-----------|----------|--------------------|
   | SVR         | 693.72    | 1080.81  | -                  |
   | Polynomial  | 110.25    | 137.58   | 84-87%             |
   | **GPR**     | **67.64** | **80.28**| **90-93%**         |

2. **Stress Distribution** (GPR Monte Carlo):
   | Statistic | Value (kPa)    |
   |-----------|----------------|
   | Mean      | 5312.27        |
   | Std       | 610.02         |
   | Median    | 5256.46        |
   | 95% CI    | [4267, 6648]   |

3. **Insights**: Right-skewed (lognormal preferred, ΔAIC=2); GPR residuals tight around zero.

## Tools and Libraries
Analysis in **Python 3.12**:
- `pandas`, `numpy`: Data handling
- `matplotlib`: Visualization
- `scikit-learn`: Surrogates (SVR, Polynomial, GPR)
- **Kratos Multiphysics**: FEM simulations

## Summary
- Built surrogates for solar sail stress prediction; **GPR best** (MSE_test=80 kPa, 93% better than SVR).
- 200 OLH simulations → noise-free dataset; train/test split shows excellent GPR fit.
- Monte Carlo UQ: Stress ~ lognormal (mean 5312 kPa, std 610 kPa); enables efficient probabilistic design vs full FEM.
