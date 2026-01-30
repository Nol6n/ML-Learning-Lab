import numpy as np
import pandas as pd
from typing import Dict, List, Any
from scipy import stats


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


def train_linear_regression(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
    **kwargs  # Accept but ignore extra parameters like train_ratio, val_ratio
) -> Dict[str, Any]:
    """
    Train linear regression with step-by-step mathematical explanations.
    Based on Chapter 3 of "An Introduction to Statistical Learning" (ISLP).

    This implementation focuses on the core concepts:
    - Least squares estimation
    - Standard errors and confidence intervals
    - Hypothesis testing (t-statistics, p-values)
    - Model fit assessment (RSE, R²)
    """
    steps = []

    # Extract data
    X_raw = df[feature_columns].values
    y = df[target_column].values
    n_samples = len(y)
    n_features = len(feature_columns)
    p = n_features + 1  # Including intercept

    # Add intercept term (column of 1s)
    X = np.column_stack([np.ones(n_samples), X_raw])

    # Calculate means for formulas
    y_mean = float(np.mean(y))
    x_means = [float(np.mean(X_raw[:, j])) for j in range(n_features)]

    # ============ STEP 1: Meet Your Data ============
    steps.append({
        "step_number": 1,
        "title": "Meet Your Data! 📊",
        "explanation": f"Welcome to linear regression! We have {n_samples} observations, and our goal is to understand the relationship between our predictor(s) and the response. Think of it like finding the best straight line through a cloud of points.",
        "latex": r"\text{We have } n = " + str(n_samples) + r" \text{ observations: } (x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)",
        "sub_steps": [
            {
                "description": f"Our response variable (what we're predicting): {target_column}",
                "latex": r"\bar{y} = " + f"{y_mean:.4f}" + r" \text{ (average value)}",
                "note": "This is what we want to predict!"
            },
            {
                "description": f"Our predictor variable(s): {', '.join(feature_columns)}",
                "latex": None if n_features > 1 else r"\bar{x} = " + f"{x_means[0]:.4f}",
                "note": "These help us make predictions"
            },
            {
                "description": "Quick peek at the data statistics:",
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

    # ============ STEP 2: The Linear Model ============
    if n_features == 1:
        model_latex = r"Y = \beta_0 + \beta_1 X + \epsilon"
        model_desc = f"We're assuming {target_column} is approximately a linear function of {feature_columns[0]}, plus some random error ε."
    else:
        terms = " + ".join([rf"\beta_{j} X_{j}" for j in range(1, n_features + 1)])
        model_latex = rf"Y = \beta_0 + {terms} + \epsilon"
        model_desc = f"We're assuming {target_column} is approximately a linear function of our {n_features} predictors, plus some random error ε."

    steps.append({
        "step_number": 2,
        "title": "The Linear Model Assumption 📐",
        "explanation": model_desc,
        "latex": model_latex,
        "sub_steps": [
            {
                "description": "β₀ (beta-zero) is the intercept — the predicted value of Y when all X's equal zero. It's where our line crosses the Y-axis!",
                "latex": r"\beta_0 = \text{intercept (the starting point)}"
            },
            {
                "description": "β₁, β₂, ... are the slopes — they tell us how much Y changes when each X increases by one unit. This is the key insight!",
                "latex": r"\beta_j = \text{the effect of a one-unit increase in } X_j \text{ on } Y"
            },
            {
                "description": "ε (epsilon) is the error term — it captures everything our simple model can't explain. No model is perfect!",
                "latex": r"\epsilon \sim \text{random noise with mean } 0"
            },
            {
                "description": "In practice, we don't know the true β values. We'll estimate them from data and call our estimates β̂ (beta-hat).",
                "latex": r"\hat{\beta} = \text{our best guess for } \beta"
            }
        ]
    })

    # ============ STEP 3: The Least Squares Idea ============
    steps.append({
        "step_number": 3,
        "title": "Finding the Best Fit: Least Squares 🎯",
        "explanation": "How do we find the 'best' line? We want to minimize the prediction errors! For each data point, we calculate the residual (the difference between actual and predicted), square it (to make all errors positive), and add them up. This sum is called RSS (Residual Sum of Squares).",
        "latex": r"\text{RSS}(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \left(y_i - \beta_0 - \beta_1 x_{i}\right)^2",
        "sub_steps": [
            {
                "description": "The residual for observation i is the prediction error:",
                "latex": r"e_i = y_i - \hat{y}_i = \text{actual} - \text{predicted}"
            },
            {
                "description": "Why square the residuals? Two reasons: (1) positive and negative errors don't cancel out, (2) larger errors get penalized more heavily.",
                "latex": r"e_i^2 \geq 0 \text{ always}"
            },
            {
                "description": "Our goal: find the β₀ and β₁ values that make RSS as small as possible. This is an optimization problem!",
                "latex": r"\min_{\beta_0, \beta_1} \text{RSS}(\beta_0, \beta_1)"
            },
            {
                "description": "To minimize RSS, we use calculus: take partial derivatives with respect to β₀ and β₁, then set them equal to zero.",
                "latex": r"\frac{\partial \text{RSS}}{\partial \beta_0} = 0 \quad \text{and} \quad \frac{\partial \text{RSS}}{\partial \beta_1} = 0"
            }
        ]
    })

    # ============ STEP 4: The Solution (Calculus Magic) ============
    if n_features == 1:
        # Simple linear regression formulas
        x_vals = X_raw.flatten()
        x_bar = float(np.mean(x_vals))

        # Calculate intermediate values for showing work
        x_minus_xbar = x_vals - x_bar
        y_minus_ybar = y - y_mean
        numerator = np.sum(x_minus_xbar * y_minus_ybar)
        denominator = np.sum(x_minus_xbar ** 2)

        # Show a few example calculations
        n_examples = min(3, n_samples)
        example_terms_num = [f"({x_vals[i]:.2f} - {x_bar:.2f})({y[i]:.2f} - {y_mean:.2f})" for i in range(n_examples)]
        example_terms_den = [f"({x_vals[i]:.2f} - {x_bar:.2f})^2" for i in range(n_examples)]

        solution_sub_steps = [
            {
                "description": "Let's derive the formula for β̂₁. Starting with RSS and taking the partial derivative with respect to β₁:",
                "latex": r"\frac{\partial}{\partial \beta_1} \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2 = -2\sum_{i=1}^{n}x_i(y_i - \beta_0 - \beta_1 x_i) = 0"
            },
            {
                "description": "Similarly, the partial derivative with respect to β₀:",
                "latex": r"\frac{\partial}{\partial \beta_0} \sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i)^2 = -2\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_i) = 0"
            },
            {
                "description": "From the second equation, we can show that β̂₀ = ȳ - β̂₁x̄. Substituting into the first equation and solving for β̂₁ gives us:",
                "latex": r"\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}"
            },
            {
                "description": f"Now let's compute the numerator. We need Σ(xᵢ - x̄)(yᵢ - ȳ) where x̄ = {x_bar:.4f} and ȳ = {y_mean:.4f}:",
                "latex": rf"\sum_{{i=1}}^{{{n_samples}}}(x_i - \bar{{x}})(y_i - \bar{{y}}) = {' + '.join(example_terms_num)} + \cdots = {numerator:.4f}"
            },
            {
                "description": f"And the denominator Σ(xᵢ - x̄)²:",
                "latex": rf"\sum_{{i=1}}^{{{n_samples}}}(x_i - \bar{{x}})^2 = {' + '.join(example_terms_den)} + \cdots = {denominator:.4f}"
            },
            {
                "description": "Dividing gives us β̂₁:",
                "latex": rf"\hat{{\beta}}_1 = \frac{{{numerator:.4f}}}{{{denominator:.4f}}} = {numerator/denominator:.6f}"
            },
            {
                "description": "Finally, we compute β̂₀ using the formula β̂₀ = ȳ - β̂₁x̄:",
                "latex": rf"\hat{{\beta}}_0 = {y_mean:.4f} - ({numerator/denominator:.6f}) \times ({x_bar:.4f}) = {y_mean - (numerator/denominator)*x_bar:.6f}"
            }
        ]
    else:
        # Multiple regression - matrix form
        XtX = X.T @ X
        XtX_inv_temp = np.linalg.inv(XtX)
        Xty = X.T @ y.reshape(-1, 1)
        beta_temp = XtX_inv_temp @ Xty

        # Build the matrix multiplication explanation row by row for final beta computation
        mult_steps = []
        coef_names_temp = ["(intercept)"] + [f"({col})" for col in feature_columns]
        for i in range(p):
            row_terms = []
            for j in range(p):
                row_terms.append(f"({XtX_inv_temp[i,j]:.4f})({Xty[j,0]:.4f})")
            row_sum = " + ".join(row_terms)
            mult_steps.append({
                "description": f"Computing β̂_{i} {coef_names_temp[i]}:",
                "latex": rf"\hat{{\beta}}_{i} = {row_sum} = {beta_temp[i,0]:.6f}"
            })

        # Build XᵀX element computation explanations
        col_names = ["1 (intercept)"] + feature_columns
        XtX_element_steps = []

        # Show a few key elements of XᵀX with their computations
        # (0,0): sum of 1*1 = n
        XtX_element_steps.append({
            "description": f"Element (1,1): The dot product of the intercept column with itself (all 1s):",
            "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})_{{11}} = \sum_{{i=1}}^{{{n_samples}}} 1 \cdot 1 = {n_samples} = {XtX[0,0]:.4f}"
        })

        # (0,1): sum of 1*x_i1 = sum of first feature
        if n_features >= 1:
            sum_x1 = np.sum(X[:, 1])
            XtX_element_steps.append({
                "description": f"Element (1,2): The dot product of intercept column with {feature_columns[0]}:",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})_{{12}} = \sum_{{i=1}}^{{{n_samples}}} 1 \cdot x_{{i1}} = \sum x_{{i1}} = {sum_x1:.4f}"
            })

        # (1,1): sum of x_i1^2
        if n_features >= 1:
            sum_x1_sq = np.sum(X[:, 1]**2)
            XtX_element_steps.append({
                "description": f"Element (2,2): The dot product of {feature_columns[0]} with itself:",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})_{{22}} = \sum_{{i=1}}^{{{n_samples}}} x_{{i1}} \cdot x_{{i1}} = \sum x_{{i1}}^2 = {sum_x1_sq:.4f}"
            })

        # (1,2): sum of x_i1 * x_i2 (if we have 2+ features)
        if n_features >= 2:
            sum_x1x2 = np.sum(X[:, 1] * X[:, 2])
            XtX_element_steps.append({
                "description": f"Element (2,3): The dot product of {feature_columns[0]} with {feature_columns[1]}:",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})_{{23}} = \sum_{{i=1}}^{{{n_samples}}} x_{{i1}} \cdot x_{{i2}} = {sum_x1x2:.4f}"
            })

        # Build Xᵀy element computation explanations
        Xty_element_steps = []

        # First element: sum of y
        sum_y = np.sum(y)
        Xty_element_steps.append({
            "description": "Element 1: The dot product of intercept column (all 1s) with y:",
            "latex": rf"(\mathbf{{X}}^T\mathbf{{y}})_1 = \sum_{{i=1}}^{{{n_samples}}} 1 \cdot y_i = \sum y_i = {sum_y:.4f}"
        })

        # Second element: sum of x_i1 * y_i
        if n_features >= 1:
            sum_x1y = np.sum(X[:, 1] * y)
            Xty_element_steps.append({
                "description": f"Element 2: The dot product of {feature_columns[0]} with y:",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{y}})_2 = \sum_{{i=1}}^{{{n_samples}}} x_{{i1}} \cdot y_i = {sum_x1y:.4f}"
            })

        # Build (XᵀX)⁻¹ computation explanation
        inverse_steps = []
        det_XtX = np.linalg.det(XtX)

        if p == 2:
            # 2x2 case - show the simple formula
            a, b = XtX[0, 0], XtX[0, 1]
            c, d = XtX[1, 0], XtX[1, 1]
            inverse_steps.append({
                "description": "For a 2×2 matrix, the inverse has a simple closed form:",
                "latex": r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}"
            })
            inverse_steps.append({
                "description": "First, compute the determinant det(XᵀX) = ad - bc:",
                "latex": rf"\det(\mathbf{{X}}^T\mathbf{{X}}) = ({a:.4f})({d:.4f}) - ({b:.4f})({c:.4f}) = {a*d:.4f} - {b*c:.4f} = {det_XtX:.4f}"
            })
            inverse_steps.append({
                "description": "Now apply the formula (swap diagonal, negate off-diagonal, divide by determinant):",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})^{{-1}} = \frac{{1}}{{{det_XtX:.4f}}} \begin{{bmatrix}} {d:.4f} & {-b:.4f} \\ {-c:.4f} & {a:.4f} \end{{bmatrix}}"
            })
        else:
            # 3x3 or larger - show cofactor method
            inverse_steps.append({
                "description": f"For a {p}×{p} matrix, we use cofactor expansion. First, compute the determinant:",
                "latex": rf"\det(\mathbf{{X}}^T\mathbf{{X}}) = {det_XtX:.4f}"
            })

            # Compute cofactor matrix
            cofactor_matrix = np.zeros_like(XtX)
            for i in range(p):
                for j in range(p):
                    # Minor: delete row i and column j
                    minor = np.delete(np.delete(XtX, i, axis=0), j, axis=1)
                    cofactor_matrix[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

            # Show a couple cofactor calculations
            if p == 3:
                # Show C_11 (cofactor of element at position 0,0)
                minor_00 = np.array([[XtX[1,1], XtX[1,2]], [XtX[2,1], XtX[2,2]]])
                det_minor_00 = XtX[1,1]*XtX[2,2] - XtX[1,2]*XtX[2,1]
                inverse_steps.append({
                    "description": "The cofactor Cᵢⱼ = (-1)^(i+j) × det(minor matrix). For example, C₁₁:",
                    "latex": rf"C_{{11}} = (-1)^{{1+1}} \det\begin{{bmatrix}} {XtX[1,1]:.4f} & {XtX[1,2]:.4f} \\ {XtX[2,1]:.4f} & {XtX[2,2]:.4f} \end{{bmatrix}} = ({XtX[1,1]:.4f})({XtX[2,2]:.4f}) - ({XtX[1,2]:.4f})({XtX[2,1]:.4f}) = {det_minor_00:.4f}"
                })

                # Show C_12
                minor_01 = np.array([[XtX[1,0], XtX[1,2]], [XtX[2,0], XtX[2,2]]])
                det_minor_01 = XtX[1,0]*XtX[2,2] - XtX[1,2]*XtX[2,0]
                inverse_steps.append({
                    "description": "Similarly, C₁₂ (note the negative sign from (-1)^(1+2)):",
                    "latex": rf"C_{{12}} = (-1)^{{1+2}} \det\begin{{bmatrix}} {XtX[1,0]:.4f} & {XtX[1,2]:.4f} \\ {XtX[2,0]:.4f} & {XtX[2,2]:.4f} \end{{bmatrix}} = -({det_minor_01:.4f}) = {-det_minor_01:.4f}"
                })

            inverse_steps.append({
                "description": "The cofactor matrix (computing all Cᵢⱼ):",
                "latex": matrix_to_latex(cofactor_matrix, r"\mathbf{C}", precision=4)
            })

            # Adjugate is transpose of cofactor
            adjugate = cofactor_matrix.T
            inverse_steps.append({
                "description": "The adjugate is the transpose of the cofactor matrix:",
                "latex": matrix_to_latex(adjugate, r"\text{adj}(\mathbf{X}^T\mathbf{X}) = \mathbf{C}^T", precision=4)
            })

            inverse_steps.append({
                "description": "Finally, divide by the determinant to get the inverse:",
                "latex": rf"(\mathbf{{X}}^T\mathbf{{X}})^{{-1}} = \frac{{1}}{{{det_XtX:.4f}}} \times \text{{adj}}(\mathbf{{X}}^T\mathbf{{X}})"
            })

        # Build the XᵀX and Xᵀy dropdown content
        XtX_dropdown_steps = [
            {
                "description": f"XᵀX is a {p}×{p} matrix where each element (i,j) is the dot product of column i and column j of X:",
                "latex": r"(\mathbf{X}^T\mathbf{X})_{ij} = \sum_{k=1}^{n} X_{ki} \cdot X_{kj} = \text{(column } i \text{)} \cdot \text{(column } j \text{)}"
            },
            *XtX_element_steps,
            {
                "description": "Computing all elements gives us the full XᵀX matrix:",
                "latex": matrix_to_latex(XtX, r"\mathbf{X}^T\mathbf{X}")
            },
            {
                "description": f"Similarly, Xᵀy is a {p}×1 vector where each element is the dot product of a column of X with y:",
                "latex": r"(\mathbf{X}^T\mathbf{y})_{i} = \sum_{k=1}^{n} X_{ki} \cdot y_k"
            },
            *Xty_element_steps,
            {
                "description": "The full Xᵀy vector:",
                "latex": matrix_to_latex(Xty, r"\mathbf{X}^T\mathbf{y}")
            }
        ]

        # Build the inverse dropdown content
        inverse_dropdown_steps = [
            {
                "description": "The matrix inverse is computed using the formula:",
                "latex": r"\mathbf{A}^{-1} = \frac{1}{\det(\mathbf{A})} \cdot \text{adj}(\mathbf{A})"
            },
            {
                "description": "Where adj(A) is the adjugate matrix (transpose of the cofactor matrix). Let's compute step by step.",
                "latex": None
            },
            *inverse_steps,
            {
                "description": "The complete (XᵀX)⁻¹ matrix:",
                "latex": matrix_to_latex(XtX_inv_temp, r"(\mathbf{X}^T\mathbf{X})^{-1}", precision=6)
            }
        ]

        solution_sub_steps = [
            {
                "description": "For multiple predictors, we use matrix notation. First, let's write our model as Y = Xβ + ε where X is the design matrix:",
                "latex": r"\mathbf{X} = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ 1 & x_{21} & \cdots & x_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np} \end{bmatrix}"
            },
            {
                "description": "RSS in matrix form becomes RSS = (y - Xβ)ᵀ(y - Xβ). Taking the derivative:",
                "latex": r"\frac{\partial \text{RSS}}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) = 0"
            },
            {
                "description": "Solving for β gives us the Normal Equations. Rearranging:",
                "latex": r"\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}"
            },
            {
                "description": "If XᵀX is invertible, we multiply both sides by (XᵀX)⁻¹:",
                "latex": r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}"
            },
            {
                "type": "dropdown",
                "title": "Computing XᵀX and Xᵀy (click to expand)",
                "description": "Let's compute XᵀX and Xᵀy step by step from our data.",
                "sub_steps": XtX_dropdown_steps
            },
            {
                "type": "dropdown",
                "title": "Computing (XᵀX)⁻¹ (click to expand)",
                "description": "Now let's compute the matrix inverse using cofactors and determinants.",
                "sub_steps": inverse_dropdown_steps
            },
            {
                "description": "Finally, we multiply (XᵀX)⁻¹ × (Xᵀy). Each element of β̂ is the dot product of a row of (XᵀX)⁻¹ with the vector Xᵀy:",
                "latex": r"\hat{\beta}_i = \sum_{j} [(\mathbf{X}^T\mathbf{X})^{-1}]_{ij} \cdot [\mathbf{X}^T\mathbf{y}]_j"
            },
            *mult_steps,
            {
                "description": "Putting it all together:",
                "latex": matrix_to_latex(beta_temp.flatten(), r"\hat{\boldsymbol{\beta}}", precision=6)
            }
        ]

    steps.append({
        "step_number": 4,
        "title": "Solving for β̂: The Calculus ✨",
        "explanation": "Now for the math! We minimize RSS by taking partial derivatives with respect to each β and setting them to zero. This gives us a system of equations we can solve directly.",
        "latex": r"\frac{\partial \text{RSS}}{\partial \beta_j} = 0 \quad \text{for each } j",
        "sub_steps": solution_sub_steps
    })

    # ============ Actually compute the solution ============
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta = XtX_inv @ Xty

    # Predictions and residuals
    y_pred = X @ beta
    residuals = y - y_pred
    RSS = float(np.sum(residuals ** 2))
    TSS = float(np.sum((y - y_mean) ** 2))

    # ============ STEP 5: Our Coefficients! ============
    coef_names = ["β̂₀ (intercept)"] + [f"β̂_{j} ({col})" for j, col in enumerate(feature_columns, 1)]

    coef_interpretations = []
    for i, (name, value) in enumerate(zip(coef_names, beta)):
        if i == 0:
            coef_interpretations.append({
                "name": name,
                "value": round(float(value), 4),
                "interpretation": f"When all predictors are 0, we predict {target_column} = {value:.4f}"
            })
        else:
            direction = "increases" if value > 0 else "decreases"
            coef_interpretations.append({
                "name": name,
                "value": round(float(value), 4),
                "interpretation": f"For each 1-unit increase in {feature_columns[i-1]}, {target_column} {direction} by {abs(value):.4f} (holding other variables constant)"
            })

    if n_features == 1:
        final_eq = rf"\hat{{y}} = {beta[0]:.4f} + {beta[1]:.4f} \cdot x"
    else:
        terms = " + ".join([f"{beta[j+1]:.4f} \\cdot x_{j+1}" for j in range(n_features)])
        final_eq = rf"\hat{{y}} = {beta[0]:.4f} + {terms}"

    steps.append({
        "step_number": 5,
        "title": "Our Coefficient Estimates! 🎉",
        "explanation": "Here they are — the β̂ values that minimize RSS. These are our best guesses for the true (unknown) population parameters.",
        "latex": final_eq,
        "sub_steps": [
            {
                "description": "The coefficient values:",
                "latex": matrix_to_latex(beta, r"\hat{\boldsymbol{\beta}}"),
                "note": f"Shape: ({p} × 1) — one coefficient for each predictor plus the intercept"
            },
            {
                "description": "What do these numbers mean?",
                "values": coef_interpretations
            }
        ]
    })

    # ============ STEP 6: How Good Are Our Estimates? ============
    # Calculate statistics for inference
    df_residual = n_samples - p  # degrees of freedom
    sigma_squared = RSS / df_residual  # estimated variance of errors
    sigma = np.sqrt(sigma_squared)  # RSE

    # Standard errors of coefficients
    var_beta = sigma_squared * XtX_inv
    se_beta = np.sqrt(np.diag(var_beta))

    # t-statistics and p-values
    t_stats = beta / se_beta
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df_residual))

    # 95% confidence intervals
    t_critical = stats.t.ppf(0.975, df_residual)
    ci_lower = beta - t_critical * se_beta
    ci_upper = beta + t_critical * se_beta

    coef_stats_table = []
    for i in range(p):
        name = "Intercept" if i == 0 else feature_columns[i-1]
        coef_stats_table.append({
            "name": name,
            "coef": round(float(beta[i]), 4),
            "se": round(float(se_beta[i]), 4),
            "t": round(float(t_stats[i]), 2),
            "p": f"< 0.0001" if p_values[i] < 0.0001 else f"{p_values[i]:.4f}",
            "ci_low": round(float(ci_lower[i]), 4),
            "ci_high": round(float(ci_upper[i]), 4)
        })

    # Build detailed SE calculation examples
    se_examples = []
    for i in range(min(2, p)):
        name = "intercept" if i == 0 else feature_columns[i-1]
        se_examples.append({
            "description": f"For {name}: SE(β̂_{i}) = √(σ² × [(XᵀX)⁻¹]_{{{i},{i}}}) = √({sigma_squared:.4f} × {XtX_inv[i,i]:.6f}) = {se_beta[i]:.4f}",
            "latex": None
        })

    # Build t-statistic examples
    t_examples = []
    for i in range(min(2, p)):
        name = "intercept" if i == 0 else feature_columns[i-1]
        t_examples.append({
            "description": f"For {name}: t = {beta[i]:.4f} / {se_beta[i]:.4f} = {t_stats[i]:.2f}",
            "latex": None
        })

    steps.append({
        "step_number": 6,
        "title": "How Confident Are We? Standard Errors & Tests 📊",
        "explanation": "Our β̂ estimates are just estimates — they'd be slightly different if we had different data. Standard errors quantify this uncertainty. We can then test whether each coefficient is significantly different from zero.",
        "latex": None,
        "sub_steps": [
            {
                "description": "First, we estimate σ² (the variance of the errors). We use the residuals from our model:",
                "latex": rf"\hat{{\sigma}}^2 = \frac{{\text{{RSS}}}}{{n - p - 1}} = \frac{{\sum_{{i=1}}^{{n}} e_i^2}}{{n - p - 1}} = \frac{{{RSS:.4f}}}{{{n_samples} - {n_features} - 1}} = {sigma_squared:.4f}"
            },
            {
                "description": f"The Residual Standard Error (RSE) is σ̂ = √{sigma_squared:.4f} = {sigma:.4f}. This is roughly how far our predictions typically miss.",
                "latex": rf"\text{{RSE}} = \hat{{\sigma}} = \sqrt{{{sigma_squared:.4f}}} = {sigma:.4f}"
            },
            {
                "description": "The variance-covariance matrix of β̂ is Var(β̂) = σ²(XᵀX)⁻¹. The standard errors are the square roots of the diagonal elements:",
                "latex": r"\text{SE}(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 \cdot [(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}"
            },
            {
                "description": "Computing (XᵀX)⁻¹ for our data:",
                "latex": matrix_to_latex(XtX_inv, r"(\mathbf{X}^T\mathbf{X})^{-1}", precision=6)
            },
            {
                "description": f"Now we calculate each standard error. For example, SE(β̂₀) = √({sigma_squared:.4f} × {XtX_inv[0,0]:.6f}) = {se_beta[0]:.4f}",
                "latex": None,
                "note": "Each SE comes from multiplying σ² by the corresponding diagonal element of (XᵀX)⁻¹"
            },
            {
                "description": "To test H₀: βⱼ = 0 (no relationship), we compute a t-statistic. Under H₀, this follows a t-distribution with n-p-1 degrees of freedom:",
                "latex": rf"t = \frac{{\hat{{\beta}}_j - 0}}{{\text{{SE}}(\hat{{\beta}}_j)}} \sim t_{{{df_residual}}}"
            },
            {
                "description": f"For example, testing if the intercept differs from 0: t = {beta[0]:.4f} / {se_beta[0]:.4f} = {t_stats[0]:.2f}",
                "latex": None
            },
            {
                "description": f"The p-value is P(|t| > |observed t|). Small p-values (< 0.05) suggest we can reject H₀. With {df_residual} degrees of freedom, the critical t-value for 95% confidence is:",
                "latex": rf"t_{{0.975, {df_residual}}} = {t_critical:.4f}"
            },
            {
                "description": "95% confidence interval: β̂ⱼ ± t₀.₉₇₅ × SE(β̂ⱼ). There's a 95% chance this interval contains the true β.",
                "latex": rf"\text{{CI}}_{{95\%}} = \hat{{\beta}}_j \pm {t_critical:.3f} \times \text{{SE}}(\hat{{\beta}}_j)"
            },
            {
                "description": "Here's the complete coefficient table with all the statistics:",
                "latex": None,
                "values": {"coefficients": coef_stats_table}
            }
        ]
    })

    # ============ STEP 7: Model Fit - R² and RSE ============
    R2 = 1 - (RSS / TSS) if TSS != 0 else 0

    # F-statistic for overall model significance
    if n_features > 0:
        F_stat = ((TSS - RSS) / n_features) / (RSS / df_residual)
        F_pvalue = 1 - stats.f.cdf(F_stat, n_features, df_residual)
    else:
        F_stat = 0
        F_pvalue = 1

    steps.append({
        "step_number": 7,
        "title": "How Well Does Our Model Fit? 📈",
        "explanation": "We've found the best line, but how good is 'best'? Two key measures: RSE tells us the typical prediction error in the units of Y, and R² tells us the proportion of variance explained (0 to 1, higher is better!).",
        "latex": r"R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = \frac{\text{variance explained}}{\text{total variance}}",
        "sub_steps": [
            {
                "description": "Total Sum of Squares (TSS) measures total variance in Y:",
                "latex": rf"\text{{TSS}} = \sum_{{i=1}}^{{n}}(y_i - \bar{{y}})^2 = {TSS:.4f}"
            },
            {
                "description": "Residual Sum of Squares (RSS) is the variance NOT explained by our model:",
                "latex": rf"\text{{RSS}} = \sum_{{i=1}}^{{n}}(y_i - \hat{{y}}_i)^2 = {RSS:.4f}"
            },
            {
                "description": f"R² (coefficient of determination) — our model explains {R2*100:.1f}% of the variance in {target_column}!",
                "latex": rf"R^2 = 1 - \frac{{{RSS:.4f}}}{{{TSS:.4f}}} = {R2:.4f}"
            },
            {
                "description": f"RSE (Residual Standard Error) — on average, our predictions are off by about {sigma:.2f} units:",
                "latex": rf"\text{{RSE}} = {sigma:.4f}"
            },
            {
                "description": "F-statistic tests if ANY of our predictors are useful (H₀: all β = 0):",
                "latex": rf"F = \frac{{(\text{{TSS}} - \text{{RSS}})/p}}{{\text{{RSS}}/(n-p-1)}} = {F_stat:.2f}, \quad p\text{{-value}} " + (r"< 0.0001" if F_pvalue < 0.0001 else rf"= {F_pvalue:.4f}"),
                "note": "Small p-value → strong evidence that at least one predictor is useful!"
            }
        ]
    })

    # ============ STEP 8: Sanity Check with sklearn ============
    from sklearn.linear_model import LinearRegression as SklearnLR

    sklearn_model = SklearnLR()
    sklearn_model.fit(X_raw, y)
    sklearn_intercept = sklearn_model.intercept_
    sklearn_coefs = sklearn_model.coef_
    sklearn_r2 = sklearn_model.score(X_raw, y)

    # Check if our results match
    intercept_match = np.isclose(beta[0], sklearn_intercept, rtol=1e-5)
    coefs_match = np.allclose(beta[1:], sklearn_coefs, rtol=1e-5)
    r2_match = np.isclose(R2, sklearn_r2, rtol=1e-5)
    all_match = intercept_match and coefs_match and r2_match

    steps.append({
        "step_number": 8,
        "title": "Sanity Check: Does sklearn Agree? ✅",
        "explanation": "Let's verify our hand calculations using Python's sklearn library. If we did the math right, the numbers should match exactly!",
        "latex": None,
        "sub_steps": [
            {
                "description": "Our hand-calculated intercept vs sklearn:",
                "latex": rf"\hat{{\beta}}_0^{{\text{{ours}}}} = {beta[0]:.6f} \quad \text{{vs}} \quad \hat{{\beta}}_0^{{\text{{sklearn}}}} = {sklearn_intercept:.6f}",
                "note": "✓ Match!" if intercept_match else "✗ Mismatch!"
            },
            {
                "description": "Our coefficients vs sklearn:",
                "latex": None,
                "note": "✓ All coefficients match!" if coefs_match else "✗ Some coefficients don't match!"
            },
            {
                "description": "Our R² vs sklearn:",
                "latex": rf"R^2_{{\text{{ours}}}} = {R2:.6f} \quad \text{{vs}} \quad R^2_{{\text{{sklearn}}}} = {sklearn_r2:.6f}",
                "note": "✓ Match!" if r2_match else "✗ Mismatch!"
            },
            {
                "description": "🎉 VERDICT: " + ("All calculations verified! Our math is correct." if all_match else "Something's off — double-check the calculations."),
                "latex": None,
                "note": "sklearn uses the same least squares method under the hood"
            }
        ]
    })

    # Prepare visualization data (single set since no validation split)
    viz_data = {
        "actual": y.tolist(),
        "predicted": y_pred.tolist(),
        "residuals": residuals.tolist(),
        "feature_values": X_raw.flatten().tolist() if n_features == 1 else X_raw[:, 0].tolist(),
    }

    if n_features == 1:
        x_min, x_max = float(X_raw.min()), float(X_raw.max())
        line_x = [x_min, x_max]
        line_y = [float(beta[0] + beta[1] * x_min), float(beta[0] + beta[1] * x_max)]
        viz_data["regression_line"] = {"x": line_x, "y": line_y}

    # Prepare coefficient statistics for display
    coef_table = {
        "headers": ["", "Coefficient", "Std. Error", "t-statistic", "p-value", "95% CI"],
        "rows": [
            {
                "name": "Intercept" if i == 0 else feature_columns[i-1],
                "coef": round(float(beta[i]), 4),
                "se": round(float(se_beta[i]), 4),
                "t": round(float(t_stats[i]), 2),
                "p": "< 0.0001" if p_values[i] < 0.0001 else f"{p_values[i]:.4f}",
                "ci": f"[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]"
            }
            for i in range(p)
        ]
    }

    return {
        "steps": steps,
        "coefficients": {
            "names": coef_names,
            "values": format_matrix(beta),
            "standard_errors": format_matrix(se_beta),
            "t_statistics": format_matrix(t_stats),
            "p_values": [float(pv) for pv in p_values],
            "confidence_intervals": {
                "lower": format_matrix(ci_lower),
                "upper": format_matrix(ci_upper)
            }
        },
        "coefficient_table": coef_table,
        "metrics": {
            "r2": round(R2, 4),
            "rse": round(sigma, 4),
            "rss": round(RSS, 4),
            "tss": round(TSS, 4),
            "f_statistic": round(F_stat, 2),
            "f_pvalue": F_pvalue
        },
        "sample_size": n_samples,
        "num_predictors": n_features,
        "degrees_of_freedom": df_residual,
        "visualization_data": viz_data,
        "sklearn_verification": {
            "intercept": float(sklearn_intercept),
            "coefficients": sklearn_coefs.tolist(),
            "r2": float(sklearn_r2),
            "all_match": bool(all_match)
        }
    }
