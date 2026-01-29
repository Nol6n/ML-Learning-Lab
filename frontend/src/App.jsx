import { useState } from 'react';
import axios from 'axios';
import FileUpload from './components/FileUpload';
import DataPreview from './components/DataPreview';
import ModelSelector from './components/ModelSelector';
import StepByStep from './components/StepByStep';
import Visualization from './components/Visualization';

const API_URL = 'http://localhost:8000';

function App() {
  const [uploadedData, setUploadedData] = useState(null);
  const [trainingResult, setTrainingResult] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(null);
  const [featureColumns, setFeatureColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');

  const handleUploadSuccess = (data) => {
    setUploadedData(data);
    setTrainingResult(null);
    setError(null);
  };

  const handleTrain = async (features, target) => {
    setIsTraining(true);
    setError(null);
    setFeatureColumns(features);
    setTargetColumn(target);

    try {
      const response = await axios.post(`${API_URL}/api/train/linear-regression`, {
        session_id: uploadedData.session_id,
        feature_columns: features,
        target_column: target,
      });
      setTrainingResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to train model');
    } finally {
      setIsTraining(false);
    }
  };

  const handleReset = () => {
    setUploadedData(null);
    setTrainingResult(null);
    setError(null);
    setFeatureColumns([]);
    setTargetColumn('');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                ML Learning Lab
              </h1>
              <p className="text-sm text-gray-500 mt-1">
                Understand machine learning, step by step
              </p>
            </div>
            {uploadedData && (
              <button
                onClick={handleReset}
                className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Start Over
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            {Array.isArray(error) ? error.join(', ') : error}
          </div>
        )}

        {/* File Upload (shown when no data uploaded) */}
        {!uploadedData && (
          <div className="py-12">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-gray-900 mb-4">
                Upload Your Dataset
              </h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Upload a CSV file and see exactly how Linear Regression works,
                with all the math explained step by step.
              </p>
            </div>
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </div>
        )}

        {/* Data Preview & Model Configuration */}
        {uploadedData && !trainingResult && (
          <div className="space-y-6">
            <DataPreview data={uploadedData} />
            <ModelSelector
              columns={uploadedData.columns}
              columnStats={uploadedData.column_stats}
              onTrain={handleTrain}
              isTraining={isTraining}
            />
          </div>
        )}

        {/* Training Results */}
        {trainingResult && (
          <div className="space-y-6">
            {/* Summary Banner */}
            <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg p-6 text-white">
              <h2 className="text-xl font-bold mb-2">
                Linear Regression Complete
              </h2>
              <p className="text-blue-100">
                Predicting <strong>{targetColumn}</strong> using{' '}
                <strong>{featureColumns.join(', ')}</strong>
              </p>
              <p className="text-blue-200 text-sm mt-1">
                Data split: {trainingResult.split_info.train_size} train / {trainingResult.split_info.val_size} validation / {trainingResult.split_info.test_size} test
              </p>
              <div className="mt-4 grid grid-cols-3 gap-4">
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-blue-200 text-xs uppercase">Train R²</div>
                  <div className="text-xl font-bold">
                    {trainingResult.metrics.train.r2}
                  </div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-blue-200 text-xs uppercase">Validation R²</div>
                  <div className="text-xl font-bold">
                    {trainingResult.metrics.validation.r2}
                  </div>
                </div>
                <div className="bg-white/10 rounded-lg p-3">
                  <div className="text-blue-200 text-xs uppercase">Test R²</div>
                  <div className="text-xl font-bold">
                    {trainingResult.metrics.test.r2}
                  </div>
                </div>
              </div>
            </div>

            {/* Step by Step Explanation */}
            <StepByStep steps={trainingResult.steps} />

            {/* Visualizations */}
            <Visualization
              result={trainingResult}
              featureColumns={featureColumns}
              targetColumn={targetColumn}
            />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            ML Learning Lab - Built for students who want to understand the math
            behind machine learning
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
