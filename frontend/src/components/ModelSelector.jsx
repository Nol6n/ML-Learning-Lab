import { useState } from 'react';

export default function ModelSelector({ columns, columnStats, onTrain, isTraining }) {
  const [featureColumns, setFeatureColumns] = useState([]);
  const [targetColumn, setTargetColumn] = useState('');

  // Filter to only show numeric columns
  const numericColumns = columns.filter(
    (col) => columnStats[col]?.type === 'numeric'
  );

  const handleFeatureToggle = (col) => {
    if (featureColumns.includes(col)) {
      setFeatureColumns(featureColumns.filter((c) => c !== col));
    } else {
      setFeatureColumns([...featureColumns, col]);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (featureColumns.length === 0 || !targetColumn) {
      return;
    }
    onTrain(featureColumns, targetColumn);
  };

  const canTrain = featureColumns.length > 0 && targetColumn && !isTraining;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">
        Configure Model
      </h2>

      <form onSubmit={handleSubmit}>
        {/* Feature Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Feature Columns (X)
          </label>
          <p className="text-sm text-gray-500 mb-3">
            Choose the column(s) you want to use to predict the target.
          </p>
          <div className="flex flex-wrap gap-2">
            {numericColumns.map((col) => (
              <button
                key={col}
                type="button"
                onClick={() => handleFeatureToggle(col)}
                disabled={col === targetColumn}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                  featureColumns.includes(col)
                    ? 'bg-blue-600 text-white'
                    : col === targetColumn
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {col}
              </button>
            ))}
          </div>
          {numericColumns.length === 0 && (
            <p className="text-amber-600 text-sm mt-2">
              No numeric columns found in your dataset.
            </p>
          )}
        </div>

        {/* Target Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Target Column (Y)
          </label>
          <p className="text-sm text-gray-500 mb-3">
            Choose the column you want to predict.
          </p>
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Select a column...</option>
            {numericColumns
              .filter((col) => !featureColumns.includes(col))
              .map((col) => (
                <option key={col} value={col}>
                  {col}
                </option>
              ))}
          </select>
        </div>

        {/* Train Button */}
        <button
          type="submit"
          disabled={!canTrain}
          className={`w-full py-3 px-4 rounded-lg font-medium text-white transition-colors ${
            canTrain
              ? 'bg-blue-600 hover:bg-blue-700'
              : 'bg-gray-300 cursor-not-allowed'
          }`}
        >
          {isTraining ? (
            <span className="flex items-center justify-center">
              <span className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></span>
              Training...
            </span>
          ) : (
            'Train Linear Regression'
          )}
        </button>
      </form>
    </div>
  );
}
