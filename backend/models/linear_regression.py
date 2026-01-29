import numpy as np
import pandas as pd
from typing import Dict, List, Any


def format_matrix(matrix: np.ndarray, precision: int = 4) -> List[List[float]]:
    """Format a numpy matrix to a list of lists with specified precision."""
    if matrix.ndim == 1:
        return [round(float(x), precision) for x in matrix]
    return [[round(float(x), precision) for x in row] for row in matrix]


def matrix_to_latex(matrix: np.ndarray, name: str = None, precision: int = 4) -> str:
    """Convert a numpy matrix to LaTeX representation."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)

    rows = []
    for row in matrix:
        row_str = " & ".join([f"{x:.{precision}f}" for x in row])
        rows.append(row_str)

    matrix_latex = r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"

    if name:
        return f"{name} = {matrix_latex}"
    return matrix_latex


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    return {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "ss_res": round(ss_res, 4),
        "ss_tot": round(ss_tot, 4)
    }


def train_linear_regression(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train linear regression with step-by-step mathematical explanations.
    Uses the Normal Equation (OLS) method with train-validation-test split.
    """
    steps = []

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Extract data
    X_raw = df[feature_columns].values
    y = df[target_column].values.reshape(-1, 1)
    n_samples = len(y)
    n_features = len(feature_columns)

    # Create shuffled indices and split
    indices = np.random.permutation(n_samples)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)

    # Split the data
    X_train_raw = X_raw[train_idx]
    X_val_raw = X_raw[val_idx]
    X_test_raw = X_raw[test_idx]

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    # Add bias term (column of 1s)
    X_train = np.column_stack([np.ones(n_train), X_train_raw])
    X_val = np.column_stack([np.ones(n_val), X_val_raw])
    X_test = np.column_stack([np.ones(n_test), X_test_raw])

    # ============ STEP 1: Data Overview ============
    steps.append({
        "step_number": 1,
        "title": "Data Overview & Train-Validation-Test Split",
        "explanation": f"We have {n_samples} data points with {n_features} feature(s). To properly evaluate our model, we split the data into three sets.",
        "latex": None,
        "sub_steps": [
            {
                "description": f"Training set ({train_ratio*100:.0f}%): {n_train} samples",
                "latex": f"n_{{train}} = {n_train}",
                "note": "Used to learn the model parameters (β values)"
            },
            {
                "description": f"Validation set ({val_ratio*100:.0f}%): {n_val} samples",
                "latex": f"n_{{val}} = {n_val}",
                "note": "Used to check for overfitting before final evaluation"
            },
            {
                "description": f"Test set ({(1-train_ratio-val_ratio)*100:.0f}%): {n_test} samples",
                "latex": f"n_{{test}} = {n_test}",
                "note": "Held out for final, unbiased performance estimate"
            },
            {
                "description": "Feature statistics (full dataset):",
                "latex": None,
                "values": {
                    col: {
                        "mean": round(float(df[col].mean()), 4),
                        "std": round(float(df[col].std()), 4),
                        "min": round(float(df[col].min()), 4),
                        "max": round(float(df[col].max()), 4)
                    }
                    for col in feature_columns + [target_column]
                }
            }
        ]
    })

    # ============ STEP 2: Problem Setup ============
    if n_features == 1:
        equation_latex = r"\hat{y} = \beta_0 + \beta_1 x"
        col_name = feature_columns[0]
    else:
        terms = " + ".join([f"\\beta_{i+1} x_{i+1}" for i in range(n_features)])
        equation_latex = f"\\hat{{y}} = \\beta_0 + {terms}"

    steps.append({
        "step_number": 2,
        "title": "Problem Setup - The Linear Model",
        "explanation": "We assume the relationship between features and target is linear. Our goal is to find the best coefficients (β values) that define this linear relationship.",
        "latex": equation_latex,
        "sub_steps": [
            {
                "description": "What each symbol means:",
                "latex": r"\hat{y} \text{ = predicted value, } \beta_0 \text{ = intercept (bias), } \beta_i \text{ = coefficient for feature } i"
            },
            {
                "description": "We can write this more compactly using matrix notation. First, we construct the design matrix X by adding a column of 1s (for the intercept):",
                "latex": r"X = \begin{bmatrix} 1 & x_{1,1} & \cdots & x_{1,p} \\ 1 & x_{2,1} & \cdots & x_{2,p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n,1} & \cdots & x_{n,p} \end{bmatrix}"
            },
            {
                "description": "Our training design matrix (first 5 rows):",
                "latex": matrix_to_latex(X_train[:min(5, n_train)], "X_{train}"),
                "note": f"Shape: ({n_train} × {n_features + 1})"
            },
            {
                "description": "Then the entire model becomes a simple matrix multiplication:",
                "latex": r"\hat{y} = X\beta \quad \text{where } \beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_p \end{bmatrix}"
            }
        ]
    })

    # ============ STEP 3: Cost Function ============
    # Calculate example errors for illustration
    y_mean = float(np.mean(y_train))
    example_indices = min(3, n_train)
    example_errors = []
    for i in range(example_indices):
        y_actual = float(y_train[i, 0])
        y_pred_zero = float(0)  # prediction when β=0
        error = y_actual - y_pred_zero
        example_errors.append({
            "actual": round(y_actual, 2),
            "predicted": round(y_pred_zero, 2),
            "error": round(error, 2),
            "error_squared": round(error**2, 2)
        })

    mse_init = float(np.mean(y_train ** 2))  # MSE when predicting 0

    steps.append({
        "step_number": 3,
        "title": "Cost Function - Measuring Model Error",
        "explanation": "To find the best β, we need a way to measure how 'wrong' our predictions are. We use Mean Squared Error (MSE) as our cost function.",
        "latex": r"J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2",
        "sub_steps": [
            {
                "description": "Breaking down the cost function:",
                "latex": r"\underbrace{(y_i - \hat{y}_i)}_{\text{error}} \rightarrow \underbrace{(y_i - \hat{y}_i)^2}_{\text{squared error}} \rightarrow \underbrace{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}_{\text{mean squared error}}"
            },
            {
                "description": "Why square the errors? (1) It penalizes large errors more than small ones. (2) It makes the math tractable (differentiable). (3) It treats over-predictions and under-predictions equally.",
                "latex": r"\text{Error of } +5 \text{ and } -5 \text{ both contribute } 25 \text{ to the cost}"
            },
            {
                "description": "In matrix form, MSE becomes:",
                "latex": r"J(\beta) = \frac{1}{n} (y - X\beta)^T (y - X\beta)"
            },
            {
                "description": "Expanding this multiplication:",
                "latex": r"J(\beta) = \frac{1}{n} \left( y^T y - 2\beta^T X^T y + \beta^T X^T X \beta \right)"
            },
            {
                "description": f"Example: With β = 0 (predicting 0 for everything), MSE = {mse_init:.4f}",
                "latex": f"J(\\mathbf{{0}}) = \\frac{{1}}{{{n_train}}} \\sum_{{i=1}}^{{{n_train}}} y_i^2 = {mse_init:.4f}"
            },
            {
                "description": "Our goal: Find β that minimizes J(β)",
                "latex": r"\beta^* = \arg\min_{\beta} J(\beta)"
            }
        ]
    })

    # ============ STEP 4: Calculus - Finding the Minimum ============
    steps.append({
        "step_number": 4,
        "title": "Calculus - Finding the Minimum",
        "explanation": "To minimize the cost function, we use calculus: take the derivative with respect to β, set it to zero, and solve for β.",
        "latex": r"\frac{\partial J}{\partial \beta} = 0",
        "sub_steps": [
            {
                "description": "Start with the expanded cost function:",
                "latex": r"J(\beta) = \frac{1}{n} \left( y^T y - 2\beta^T X^T y + \beta^T X^T X \beta \right)"
            },
            {
                "description": "Take the derivative with respect to β (using matrix calculus rules):",
                "latex": r"\frac{\partial J}{\partial \beta} = \frac{1}{n} \left( -2X^T y + 2X^T X \beta \right)"
            },
            {
                "description": "Set the derivative equal to zero:",
                "latex": r"\frac{1}{n} \left( -2X^T y + 2X^T X \beta \right) = 0"
            },
            {
                "description": "Simplify (multiply both sides by n/2):",
                "latex": r"-X^T y + X^T X \beta = 0"
            },
            {
                "description": "Rearrange to isolate β:",
                "latex": r"X^T X \beta = X^T y"
            },
            {
                "description": "Solve for β by multiplying both sides by the inverse of (X'X):",
                "latex": r"\beta = (X^T X)^{-1} X^T y"
            },
            {
                "description": "This is the Normal Equation - the closed-form solution for linear regression!",
                "latex": r"\boxed{\beta^* = (X^T X)^{-1} X^T y}"
            }
        ]
    })

    # ============ STEP 5: Computing the Solution ============
    XtX = X_train.T @ X_train
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_train.T @ y_train
    beta = XtX_inv @ Xty

    # Show matrix multiplication example for small matrices
    if n_features == 1:
        xtx_explanation = f"X^T X is a {n_features+1}×{n_features+1} matrix. Each element (i,j) is the dot product of column i and column j of X."
    else:
        xtx_explanation = f"X^T X is a {n_features+1}×{n_features+1} matrix called the Gram matrix."

    steps.append({
        "step_number": 5,
        "title": "Computing the Solution (Training Data Only)",
        "explanation": "Now we compute each part of the Normal Equation using only training data. This is crucial - we never use validation or test data for training!",
        "latex": r"\beta = (X_{train}^T X_{train})^{-1} X_{train}^T y_{train}",
        "sub_steps": [
            {
                "description": f"Step 5.1: Compute X'X (Gram Matrix) - {xtx_explanation}",
                "latex": matrix_to_latex(XtX, "X^T X")
            },
            {
                "description": "The diagonal elements represent the sum of squares of each column. Off-diagonal elements represent correlations between columns.",
                "latex": f"(X^T X)_{{00}} = \\sum_{{i=1}}^{{{n_train}}} 1^2 = {n_train} \\quad \\text{{(count of samples)}}"
            },
            {
                "description": "Step 5.2: Compute the inverse (X'X)⁻¹",
                "latex": matrix_to_latex(XtX_inv, "(X^T X)^{-1}")
            },
            {
                "description": "Step 5.3: Compute X'y - the correlation between features and target",
                "latex": matrix_to_latex(Xty, "X^T y")
            },
            {
                "description": f"The first element ({float(Xty[0,0]):.2f}) is the sum of all y values. Other elements measure how each feature correlates with y.",
                "latex": f"(X^T y)_0 = \\sum_{{i=1}}^{{{n_train}}} y_i = {float(Xty[0,0]):.2f}"
            },
            {
                "description": "Step 5.4: Final multiplication → Our coefficients!",
                "latex": matrix_to_latex(beta, r"\beta^*")
            }
        ]
    })

    # ============ STEP 6: Validation ============
    y_train_pred = X_train @ beta
    y_val_pred = X_val @ beta

    train_metrics = calculate_metrics(y_train, y_train_pred)
    val_metrics = calculate_metrics(y_val, y_val_pred)

    # Calculate example predictions for validation
    val_examples = []
    for i in range(min(3, n_val)):
        x_val = X_val_raw[i, 0] if n_features == 1 else X_val_raw[i, :]
        y_actual = float(y_val[i, 0])
        y_pred = float(y_val_pred[i, 0])
        val_examples.append({
            "x": round(float(x_val) if n_features == 1 else x_val[0], 2),
            "y_actual": round(y_actual, 2),
            "y_pred": round(y_pred, 2),
            "error": round(y_actual - y_pred, 2)
        })

    r2_drop = train_metrics["r2"] - val_metrics["r2"]
    if r2_drop > 0.1:
        overfitting_warning = f"⚠️ R² dropped significantly ({r2_drop:.3f}). The model may be overfitting to training data."
    elif r2_drop > 0.05:
        overfitting_warning = f"R² dropped moderately ({r2_drop:.3f}). Monitor for potential overfitting."
    else:
        overfitting_warning = f"✓ R² drop is small ({r2_drop:.3f}). Good generalization!"

    steps.append({
        "step_number": 6,
        "title": "Validation - Checking for Overfitting",
        "explanation": "We now test our model on the validation set - data it has never seen. If performance drops significantly, the model may be 'memorizing' training data rather than learning the true pattern.",
        "latex": r"\hat{y}_{val} = X_{val} \cdot \beta",
        "sub_steps": [
            {
                "description": "Making predictions on validation data:",
                "latex": r"\text{For each validation point: } \hat{y}_i = \beta_0 + \beta_1 x_{i,1} + \cdots + \beta_p x_{i,p}"
            },
            {
                "description": "Training Set Metrics:",
                "latex": f"R^2_{{train}} = {train_metrics['r2']}, \\quad RMSE_{{train}} = {train_metrics['rmse']}",
                "values": {
                    "set": "Training",
                    "r2": train_metrics["r2"],
                    "mse": train_metrics["mse"],
                    "rmse": train_metrics["rmse"],
                    "mae": train_metrics["mae"]
                }
            },
            {
                "description": "Validation Set Metrics:",
                "latex": f"R^2_{{val}} = {val_metrics['r2']}, \\quad RMSE_{{val}} = {val_metrics['rmse']}",
                "values": {
                    "set": "Validation",
                    "r2": val_metrics["r2"],
                    "mse": val_metrics["mse"],
                    "rmse": val_metrics["rmse"],
                    "mae": val_metrics["mae"]
                }
            },
            {
                "description": "Understanding R² (Coefficient of Determination):",
                "latex": r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}"
            },
            {
                "description": "R² interpretation: What fraction of variance in y is explained by our model? R²=1 means perfect predictions, R²=0 means no better than predicting the mean.",
                "latex": f"R^2_{{val}} = 1 - \\frac{{{val_metrics['ss_res']:.2f}}}{{{val_metrics['ss_tot']:.2f}}} = {val_metrics['r2']}"
            },
            {
                "description": overfitting_warning,
                "latex": f"\\Delta R^2 = R^2_{{train}} - R^2_{{val}} = {train_metrics['r2']} - {val_metrics['r2']} = {r2_drop:.4f}"
            }
        ]
    })

    # ============ STEP 7: Final Test Evaluation ============
    y_test_pred = X_test @ beta
    test_metrics = calculate_metrics(y_test, y_test_pred)

    coef_names = ["β₀ (intercept)"] + [f"β_{i+1} ({col})" for i, col in enumerate(feature_columns)]
    coef_interpretations = []

    for i, (name, value) in enumerate(zip(coef_names, beta.flatten())):
        if i == 0:
            coef_interpretations.append({
                "name": name,
                "value": round(float(value), 4),
                "interpretation": f"When all features are 0, the predicted {target_column} is {value:.4f}"
            })
        else:
            direction = "increases" if value > 0 else "decreases"
            coef_interpretations.append({
                "name": name,
                "value": round(float(value), 4),
                "interpretation": f"For each unit increase in {feature_columns[i-1]}, {target_column} {direction} by {abs(value):.4f}"
            })

    if n_features == 1:
        final_eq = f"\\hat{{y}} = {beta[0,0]:.4f} + {beta[1,0]:.4f} \\cdot x"
    else:
        terms = " + ".join([f"{beta[i+1,0]:.4f} \\cdot x_{i+1}" for i in range(n_features)])
        final_eq = f"\\hat{{y}} = {beta[0,0]:.4f} + {terms}"

    steps.append({
        "step_number": 7,
        "title": "Final Test Evaluation & Interpretation",
        "explanation": "Finally, we evaluate on the test set - data completely held out from both training and validation. This gives us an unbiased estimate of real-world performance.",
        "latex": final_eq,
        "sub_steps": [
            {
                "description": "Test Set Performance (final, unbiased evaluation):",
                "latex": f"R^2_{{test}} = {test_metrics['r2']}, \\quad RMSE_{{test}} = {test_metrics['rmse']}",
                "values": {
                    "set": "Test",
                    "r2": test_metrics["r2"],
                    "mse": test_metrics["mse"],
                    "rmse": test_metrics["rmse"],
                    "mae": test_metrics["mae"]
                }
            },
            {
                "description": "What the coefficients mean:",
                "values": coef_interpretations
            },
            {
                "description": f"Interpretation: Our model explains {test_metrics['r2']*100:.1f}% of the variance in {target_column}. On average, predictions are off by ±{test_metrics['rmse']:.2f} (RMSE).",
                "latex": None
            },
            {
                "description": "Performance comparison across all sets:",
                "values": {
                    "comparison": [
                        {"set": "Training", "r2": train_metrics["r2"], "mse": train_metrics["mse"], "rmse": train_metrics["rmse"]},
                        {"set": "Validation", "r2": val_metrics["r2"], "mse": val_metrics["mse"], "rmse": val_metrics["rmse"]},
                        {"set": "Test", "r2": test_metrics["r2"], "mse": test_metrics["mse"], "rmse": test_metrics["rmse"]}
                    ]
                }
            }
        ]
    })

    # Prepare visualization data
    viz_data = {
        "train": {
            "actual": y_train.flatten().tolist(),
            "predicted": y_train_pred.flatten().tolist(),
            "residuals": (y_train - y_train_pred).flatten().tolist(),
            "feature_values": X_train_raw.flatten().tolist() if n_features == 1 else X_train_raw[:, 0].tolist(),
            "indices": train_idx.tolist()
        },
        "validation": {
            "actual": y_val.flatten().tolist(),
            "predicted": y_val_pred.flatten().tolist(),
            "residuals": (y_val - y_val_pred).flatten().tolist(),
            "feature_values": X_val_raw.flatten().tolist() if n_features == 1 else X_val_raw[:, 0].tolist(),
            "indices": val_idx.tolist()
        },
        "test": {
            "actual": y_test.flatten().tolist(),
            "predicted": y_test_pred.flatten().tolist(),
            "residuals": (y_test - y_test_pred).flatten().tolist(),
            "feature_values": X_test_raw.flatten().tolist() if n_features == 1 else X_test_raw[:, 0].tolist(),
            "indices": test_idx.tolist()
        }
    }

    if n_features == 1:
        x_min, x_max = float(X_raw.min()), float(X_raw.max())
        line_x = [x_min, x_max]
        line_y = [float(beta[0] + beta[1] * x_min), float(beta[0] + beta[1] * x_max)]
        viz_data["regression_line"] = {"x": line_x, "y": line_y}

    return {
        "steps": steps,
        "coefficients": {
            "names": coef_names,
            "values": format_matrix(beta.flatten())
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics
        },
        "split_info": {
            "train_size": n_train,
            "val_size": n_val,
            "test_size": n_test,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1 - train_ratio - val_ratio
        },
        "visualization_data": viz_data
    }
