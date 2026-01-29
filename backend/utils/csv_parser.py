import pandas as pd
import numpy as np
from io import StringIO
from typing import Dict, List, Any


def parse_csv(content: str) -> pd.DataFrame:
    """Parse CSV content into a pandas DataFrame."""
    return pd.read_csv(StringIO(content))


def get_csv_preview(df: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
    """Get a preview of the CSV data with column info."""
    preview_df = df.head(max_rows)

    # Get column statistics for numeric columns
    stats = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats[col] = {
                "type": "numeric",
                "mean": round(float(df[col].mean()), 4),
                "std": round(float(df[col].std()), 4),
                "min": round(float(df[col].min()), 4),
                "max": round(float(df[col].max()), 4),
                "count": int(df[col].count())
            }
        else:
            stats[col] = {
                "type": "categorical",
                "unique": int(df[col].nunique()),
                "count": int(df[col].count())
            }

    return {
        "columns": list(df.columns),
        "preview": preview_df.replace({np.nan: None}).to_dict(orient="records"),
        "row_count": len(df),
        "column_stats": stats
    }


def validate_columns(df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
    """Validate that selected columns exist and are numeric."""
    errors = []

    all_columns = feature_columns + [target_column]
    for col in all_columns:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in dataset")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' is not numeric")

    if errors:
        return {"valid": False, "errors": errors}

    return {"valid": True, "errors": []}
