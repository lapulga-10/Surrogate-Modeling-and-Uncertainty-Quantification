import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, make_scorer

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

import seaborn as sns




# Load dataset
df = pd.read_csv("design_out.csv")

# Remove sigma_mem_y as it represents the membrane stress and is used to calculate probability of failure
# Not a model parametere which influences the model but helps to calculate the structure failure probability
df = df[['f_mem','sigma_mem','E_mem','nu_mem','sigma_edg','sigma_sup','sigma_mem_max']]



# Split into features (X) and target (y)
X = df.iloc[:, :-1]  # all columns except last
y = df.iloc[:, -1]   # last column = response

# Split the data into training and testing sets 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


## Q1 : Surrogate Model Selection and Training ============================

# 1) SVR Model
# Build pipeline 
# First standardize the data, then fit SVR model
pipeline = make_pipeline(StandardScaler(), SVR())

# Define search grid
param_grid = {
    "svr__C": [0.1, 1, 10, 100, 1000],
    "svr__epsilon": [0.001, 0.01, 0.1, 0.5, 1],
    "svr__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]  # scale : (1 / (n_features * X.var())) and auto : (1 / n_features)  
}

# Run grid search with 5-fold CV
grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Build final SVR model with optimal hyperparameters
svr = make_pipeline(
    StandardScaler(),
    SVR(kernel='rbf', C=grid.best_params_['svr__C'], epsilon=grid.best_params_['svr__epsilon'], gamma=grid.best_params_['svr__gamma'])
)

# Fit on the FULL dataset (not train/test split)
svr.fit(X_train, y_train)

y_train_pred_svr = svr.predict(X_train)
y_test_pred_svr = svr.predict(X_test)

# Training metrics
r2_train_svr = r2_score(y_train, y_train_pred_svr)
mse_train_svr = mean_squared_error(y_train, y_train_pred_svr)

# Test metrics
r2_test_svr = r2_score(y_test, y_test_pred_svr)
mse_test_svr = mean_squared_error(y_test, y_test_pred_svr)



# Create datafram for errors storage with row names as model names and columns as Train and Test errors
errors_df = pd.DataFrame(
    data={
        "R2_Train": [r2_train_svr],
        "R2_Test": [r2_test_svr],
        "MSE_Train": [mse_train_svr],
        "MSE_Test": [mse_test_svr]
    },
    index=["SVR"]
)



# Scatter plot of true vs predicted values for test set
fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background

plt.scatter(y_test, y_test_pred_svr)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("SVR Model: True vs Predicted Values on Test Set")
plt.savefig("results_2/SVR_test.png", dpi=300, bbox_inches="tight")
plt.show()


# Residuals plot for test set
residuals_test_svr = y_test - y_test_pred_svr

fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background

plt.scatter(y_test_pred_svr, residuals_test_svr)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("SVR: Residuals vs Predicted (Test)")
ax.set_ylim(-120, 120)
plt.savefig("results_2/SVR_res_test.png", dpi=300, bbox_inches="tight")
plt.show()



# 2) Polynomial Regression Model
# List of degrees to try
degrees = [1, 2, 3, 4, 5, 6]

best_degree = None
best_mse = float('inf')
mse_scores_per_degree = []

for degree in degrees:
    # Create pipeline: polynomial features + scaling + linear regression
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(fit_intercept=True)
    )
    
    # 5-fold cross-validation for MSE (negative because cross_val_score maximizes score)
    neg_mse_scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring=make_scorer(mean_squared_error))
    mse_scores = neg_mse_scores.mean()  # Average MSE across folds
    mse_scores_per_degree.append(mse_scores)
    
    # Keep track of the best degree
    if mse_scores < best_mse:
        best_mse = mse_scores
        best_degree = degree


# Fit final model with best degree on full training data
poly = make_pipeline(
    PolynomialFeatures(degree=best_degree, include_bias=False),
    StandardScaler(),
    LinearRegression(fit_intercept=True)
)

poly.fit(X_train, y_train)

# Predictions
y_train_pred_poly = poly.predict(X_train)
y_test_pred_poly = poly.predict(X_test)

# Training and test metrics

r2_train_poly = r2_score(y_train, y_train_pred_poly)
mse_train_poly = mean_squared_error(y_train, y_train_pred_poly)
r2_test_poly = r2_score(y_test, y_test_pred_poly)
mse_test_poly = mean_squared_error(y_test, y_test_pred_poly)

# Errors from Polynomial Regression
errors_df.loc["PLY"] = [r2_train_poly, r2_test_poly, mse_train_poly, mse_test_poly]




# Scatter plot of residuals for test set
fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background


# Scatter plot of true vs predicted values
plt.scatter(y_test, y_test_pred_poly)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("PLY Model: True vs Predicted Values on Test Set")
plt.savefig("results_2/PLY_test.png", dpi=300, bbox_inches="tight")
plt.show()

# Residuals plot for test set
residuals_test_poly = y_test - y_test_pred_poly
fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background

plt.scatter(y_test_pred_poly, residuals_test_poly)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("PLY: Residuals vs Predicted (Test)")
ax.set_ylim(-120, 120)
plt.savefig("results_2/PLY_res_test.png", dpi=300, bbox_inches="tight")
plt.show()


# 3) Gaussian Process Regression Model

# Scale features
scaler = StandardScaler()

# Define kernel: Constant * RBF
# kernel = C(1.0, (1e-3, 1e6)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-3, 1e2))
# kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
# kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)




# Initialize Gaussian Process Regressor
gpr = make_pipeline(
    StandardScaler(),
    GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
)

# Fit model
gpr.fit(X_train, y_train)

# print(gpr.kernel_)
# Predictions
y_train_pred_gp, y_train_std_gp = gpr.predict(X_train, return_std=True)
y_test_pred_gp, y_test_std_gp = gpr.predict(X_test, return_std=True)

# Metrics
r2_train_gp = r2_score(y_train, y_train_pred_gp)
mse_train_gp = mean_squared_error(y_train, y_train_pred_gp)
r2_test_gp = r2_score(y_test, y_test_pred_gp)
mse_test_gp = mean_squared_error(y_test, y_test_pred_gp)


# Errors from Gaussian Regression
errors_df.loc["GPR"] = [r2_train_gp, r2_test_gp, mse_train_gp, mse_test_gp]

# Scatter plot of residuals for test set
fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background

# Scatter plot of true vs predicted values
plt.scatter(y_test, y_test_pred_gp)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("GP Model: True vs Predicted Values on Test Set")
plt.savefig("results_2/GP_test.png", dpi=300, bbox_inches="tight")
plt.show()


# Residuals plot for test set
residuals_test_gp = y_test - y_test_pred_gp

fig, ax = plt.subplots()
fig.patch.set_facecolor("#ff00002a")  # figure background

plt.scatter(y_test_pred_gp, residuals_test_gp)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.title("GP: Residuals vs Predicted (Test)")
ax.set_ylim(-120, 120)
plt.savefig("results_2/GP_res_test.png", dpi=300, bbox_inches="tight")
plt.show()





# Select only MSE columns
mse_df = errors_df[["MSE_Train", "MSE_Test"]]


# MSE plot grouped bar chart
ax = mse_df.plot(kind="bar", figsize=(6, 4))
ax.set_xlabel("Model")
ax.set_ylabel("MSE")
ax.set_title("Training and Testing MSE by Model")
plt.tight_layout()
plt.savefig("results_2/mse_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# Gives GPR as the best model based on MSE




# Q2 : Uncertainty Propagation using GPR Model =============
# Constants
N = 10000 #sample size
gamma = 0.5772156649  # Euler-Mascheroni constant for gumbel distribution

def get_lognormal_params(mean, sd):
    var = sd**2
    sigma = np.sqrt(np.log(1 + var / mean**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return mu, sigma

def get_gumbel_params(mean, sd):
    scale = sd * np.sqrt(6) / np.pi
    loc = mean - scale * gamma
    return loc, scale

def get_uniform_params(mean, sd):
    # SD = (high - low) / sqrt(12)
    # Mean = (high + low) / 2
    delta = sd * np.sqrt(3)
    low = mean - delta
    high = mean + delta
    return low, high

# Generate data
data = {}

# f_mem: Gumbel
loc, scale = get_gumbel_params(0.4, 0.12)
data['f_mem'] = np.random.gumbel(loc, scale, N)

# Lognormal variables
lognorm_vars = {
    'sigma_mem_y': (11000, 1650),
    'sigma_mem': (4000, 800),
    'E_mem': (600000, 90000),
    'sigma_edg': (353677.6513, 70735.53026),
    'sigma_sup': (400834.6715, 80166.9343)
}

for name, (m, s) in lognorm_vars.items():
    mu, sigma = get_lognormal_params(m, s)
    data[name] = np.random.lognormal(mu, sigma, N)

# nu_mem: Uniform
low, high = get_uniform_params(0.4, 0.01154700538)
data['nu_mem'] = np.random.uniform(low, high, N)

try:
    samples_df = pd.read_csv('sample_points.csv')
except FileNotFoundError:
    # Create DataFrame
    samples_df = pd.DataFrame(data)
    # Save to CSV
    samples_df.to_csv('sample_points.csv', index=False)

# Summary statistics for verification
summary = samples_df.agg(['mean', 'std']).T
summary['variance'] = summary['std']**2









samples_df = samples_df[['f_mem','sigma_mem','E_mem','nu_mem','sigma_edg','sigma_sup']]
samples = samples_df.values








def uncertainty(model, samples, model_name="Surrogate Model", unit="Units"):
    """
    Performs uncertainty quantification on a model using provided samples.
    
    Parameters:
    model: Trained sklearn-compatible model (SVR, GP, etc.)
    samples: (N, features) array of Monte Carlo samples
    model_name: String for plot titles
    unit: String for axis labels (e.g., 'MPa' or 'kN')
    """
    # 1. Generate Predictions
    predictions = model.predict(samples).flatten()
    
    # 2. Calculate Statistics
    mu = np.mean(predictions)
    sigma = np.std(predictions)
    cv = (sigma / mu) * 100 if mu != 0 else 0  # Coefficient of Variation
    
    # 3. Create Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Plot 1: Histogram & KDE ---
    sns.histplot(predictions, kde=True, ax=ax1, color='teal', bins=40, stat="density", alpha=0.4)
    ax1.axvline(mu, color='red', linestyle='--', label=f'Mean: {mu:.2f}')
    ax1.axvline(mu - 2*sigma, color='orange', linestyle=':', label=f'±2σ (95% CI)')
    ax1.axvline(mu + 2*sigma, color='orange', linestyle=':')
    
    ax1.set_title(f"Uncertainty Distribution: {model_name}")
    ax1.set_xlabel(f"Predicted Output ({unit})")
    ax1.set_xlim(3000, 9000)
    ax1.set_ylim(0, 0.001)
    ax1.legend()
    
    # --- Plot 2: Box & Whisker (Outlier Analysis) ---
    sns.boxplot(y=predictions, ax=ax2, color='lightblue', width=0.4)
    ax2.set_title("Statistical Spread")
    ax2.set_ylabel(f"Output ({unit})")
    
    plt.tight_layout()
    plt.show()
    
    # 4. Return Summary Report
    return {
        "Mean": mu,
        "Std_Dev": sigma,
        "COV_Percent": cv,
        "Min": np.min(predictions),
        "Max": np.max(predictions),
        "predictions": predictions
    }



y_hat_gpr = uncertainty_gpr['predictions']


from scipy.stats import gamma, lognorm

# --- 1) Lognormal fit (only positive values) ---
y_pos = y_hat_gpr[y_hat_gpr > 0]
log_data = np.log(y_pos)
mu_log = np.mean(log_data)
sigma_log = np.std(log_data, ddof=1)
shape = sigma_log              # shape parameter
scale = np.exp(mu_log)         # scale parameter


# 2) Gamma fit (only positive values) ---
y_pos = y_hat_gpr[y_hat_gpr > 0]
shape_gamma, loc_gamma, scale_gamma = gamma.fit(y_pos)

# --- Grid for pdfs ---
x_min = max(0, y_hat_gpr.min())
x_max = y_hat_gpr.max()
x = np.linspace(x_min, x_max, 500)


pdf_logn = lognorm.pdf(x, s=shape, scale=scale)
pdf_gamma = gamma.pdf(x, a=shape_gamma, loc=loc_gamma, scale=scale_gamma)

GPR_logn_AIC = 128272.81692383962
GPR_gamma_AIC = 128274.81549479174

plt.figure(figsize=(7,4))
plt.hist(y_hat_gpr, bins=40, density=True, alpha=0.4, color='C0',
         label='Empirical') 

plt.plot(
    x, pdf_logn, 'C2-', lw=2,
    label=fr'Lognormal fit (AIC = {GPR_logn_AIC:.1f})'
)
plt.plot(
    x, pdf_gamma, 'C3--', lw=2,
    label=fr'Gamma fit (AIC = {GPR_gamma_AIC:.1f})'
)

plt.xlabel(r'$\sigma_{\mathrm{mem,max}}$')
plt.ylabel('Density')
plt.legend()
plt.xlim(3000, 9000)
plt.ylim(0, 0.0008)
plt.tight_layout()
plt.savefig("results_2/fit_gpr.png", dpi=300, bbox_inches="tight")
plt.show()


# AIC of Lognormal
def aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

n = len(y_hat_gpr)
k_logn = 2  # mu and sigma
k_gamma = 3  # shape, loc, scale

# Lognormal RSS
rss_logn_gpr = np.sum((y_hat_gpr - lognorm.mean(s=shape, scale=scale))**2)
aic_logn_gpr = aic(n, rss_logn_gpr, k_logn)
print("GPR Lognormal AIC:", aic_logn_gpr)

# Gamma RSS
rss_gamma_gpr = np.sum((y_hat_gpr - gamma.mean(a=shape_gamma, loc=loc_gamma, scale=scale_gamma))**2)
aic_gamma_gpr = aic(n, rss_gamma_gpr, k_gamma)














