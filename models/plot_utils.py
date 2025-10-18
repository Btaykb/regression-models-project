import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


def plot_pred_vs_actual(y_test, y_pred, save_path=None, figsize=(6, 6), dpi=150):
    """
    Scatter predicted vs actual with 1:1 line and metrics in the title.
    y_test, y_pred: 1D arrays or pandas Series (same length)
    save_path: optional path to save the figure (PNG recommended)
    """
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(y_test, y_pred, alpha=0.6, s=20, edgecolor='none')
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    pad = 0.02 * (mx - mn) if mx > mn else 1.0
    plt.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
             color='red', linestyle='--', linewidth=1)
    plt.xlabel('Actual (y_test)')
    plt.ylabel('Predicted (y_pred)')
    plt.title(f'Predicted vs Actual — R²={r2:.3f}, RMSE={rmse:.3f}')
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved plot to: {save_path}")

    plt.show()
