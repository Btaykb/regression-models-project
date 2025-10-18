# Regression Models Project

A collection of regression models (Linear, Random Forest, SVR) exploring predictions on the California Housing dataset.

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/Btaykb/regression-models-project.git
cd regression-models-project
```

2. Create and activate the conda environment
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate reg-project
```

## Running the Models

All model scripts are in the `models/` directory. You can run them individually:

```bash
# From the project root directory:

# Linear Regression
python models/multi_reg_model.py

# Random Forest
python models/random_forest_model.py
python models/random_forest_tuned_model.py

# SVR (Support Vector Regression)
python models/svr_model.py
python models/svr_model_tuned.py
```

Results and plots will be saved in `models/result-plots/`.

## Project Structure
```
├── data/                   # Dataset directory
├── models/                 # Model implementations
│   ├── result-plots/      # Generated plots and visualizations
│   ├── multi_reg_model.py # Linear regression implementation
│   ├── plot_utils.py      # Plotting utilities
│   └── ...                # Other model implementations
├── environment.yml        # Conda environment specification
└── README.md
```

A small project exploring regression models on the California housing dataset (Sklearn)

## Summary of results
- Best model: Random Forest
- Key metrics:
| Model | R² (val) |
|---|---:|---:|
| Multi Regression | 0.57578 |
| Polynomial Regression | 0.64569 | 
| Random Forest | 0.80507 | 
| Random Forest (Tuned) | 0.80646 |
| Support Vector Regression | 0.72756 |
| Support Vector Regression (Tuned) | 0.75971 |

For a better view of the results, including RMSE, refer to the plots in the    `/models/result-plots/` folder.

Conclusion: Random Forest gave the best validation R². Some performance tuning yielded a sligthly better R² result. 

The better performace of Random forest could be attributed to the model performing better on non-linear relationships in data, while other less complex models such as multi and polynomial fare worse on such non-linear datasets.

## Reproducibility / How to run
1. Create the conda environment (recommended):
```bash
conda env create -f [environment.yml](http://_vscodecontentref_/5)
conda activate reg-project