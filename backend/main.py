from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from utils.csv_parser import parse_csv, get_csv_preview, validate_columns
from models.linear_regression import train_linear_regression

app = FastAPI(
    title="ML Learning Lab API",
    description="API for step-by-step machine learning explanations",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded data (in production, use proper storage)
uploaded_data = {}


class TrainRequest(BaseModel):
    session_id: str
    feature_columns: List[str]
    target_column: str


@app.get("/")
async def root():
    return {"message": "ML Learning Lab API", "status": "running"}


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file and get a preview."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        df = parse_csv(content_str)

        # Generate a session ID
        import uuid
        session_id = str(uuid.uuid4())

        # Store the dataframe
        uploaded_data[session_id] = df

        preview = get_csv_preview(df)
        preview["session_id"] = session_id
        preview["filename"] = file.filename

        return preview

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")


@app.post("/api/train/linear-regression")
async def train_linear_regression_endpoint(request: TrainRequest):
    """Train linear regression model with step-by-step explanations."""
    if request.session_id not in uploaded_data:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a CSV first.")

    df = uploaded_data[request.session_id]

    # Validate columns
    validation = validate_columns(df, request.feature_columns, request.target_column)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["errors"])

    try:
        result = train_linear_regression(
            df,
            request.feature_columns,
            request.target_column
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/api/sessions/{session_id}")
async def get_session_data(session_id: str):
    """Get data for a specific session."""
    if session_id not in uploaded_data:
        raise HTTPException(status_code=404, detail="Session not found")

    df = uploaded_data[session_id]
    return get_csv_preview(df)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data."""
    if session_id in uploaded_data:
        del uploaded_data[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")
