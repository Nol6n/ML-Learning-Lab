# ML Learning Lab

A web application for university students to upload CSV datasets and see the step-by-step mathematical workings of machine learning models, starting with Linear Regression.

## Features

- Upload CSV datasets with drag-and-drop
- Interactive data preview with column statistics
- Step-by-step mathematical explanations with LaTeX rendering
- Visualizations including scatter plots, residual plots, and actual vs predicted charts
- Full derivation of the Normal Equation for Linear Regression

## Tech Stack

- **Frontend**: React + Vite, Tailwind CSS, KaTeX, Recharts
- **Backend**: FastAPI (Python), NumPy, Pandas

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```

The app will be available at `http://localhost:5173`

## Usage

1. Open the app in your browser at `http://localhost:5173`
2. Upload a CSV file (a sample file `sample_data.csv` is provided in the root directory)
3. Select your feature column(s) (X) and target column (Y)
4. Click "Train Linear Regression"
5. Explore the step-by-step mathematical explanation

## Sample Data

A sample CSV file (`sample_data.csv`) is included with student study data:
- `hours_studied`: Hours spent studying
- `practice_tests`: Number of practice tests taken
- `previous_score`: Score on previous exam
- `final_score`: Final exam score (target variable)

## API Endpoints

- `POST /api/upload` - Upload a CSV file
- `POST /api/train/linear-regression` - Train a linear regression model
- `GET /api/sessions/{session_id}` - Get session data
- `DELETE /api/sessions/{session_id}` - Delete a session

## What You'll Learn

The app explains:
1. **Data Overview** - Basic statistics of your dataset
2. **Problem Setup** - The linear model equation and matrix notation
3. **Cost Function** - Mean Squared Error (MSE) and why we use it
4. **Normal Equation** - Step-by-step derivation with actual matrix calculations
5. **Results** - Coefficient interpretation and model evaluation metrics
