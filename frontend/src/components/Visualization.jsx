import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ReferenceLine,
  ComposedChart,
  Legend,
} from 'recharts';

export default function Visualization({ result, featureColumns, targetColumn }) {
  const { visualization_data, metrics, coefficients, split_info } = result;

  // Prepare scatter plot data for each set
  const trainData = visualization_data.train.actual.map((actual, idx) => ({
    x: visualization_data.train.feature_values[idx],
    actual,
    predicted: visualization_data.train.predicted[idx],
    residual: visualization_data.train.residuals[idx],
    set: 'Train',
  }));

  const valData = visualization_data.validation.actual.map((actual, idx) => ({
    x: visualization_data.validation.feature_values[idx],
    actual,
    predicted: visualization_data.validation.predicted[idx],
    residual: visualization_data.validation.residuals[idx],
    set: 'Validation',
  }));

  const testData = visualization_data.test.actual.map((actual, idx) => ({
    x: visualization_data.test.feature_values[idx],
    actual,
    predicted: visualization_data.test.predicted[idx],
    residual: visualization_data.test.residuals[idx],
    set: 'Test',
  }));

  // Regression line data (for single feature)
  const lineData = visualization_data.regression_line
    ? [
        { x: visualization_data.regression_line.x[0], y: visualization_data.regression_line.y[0] },
        { x: visualization_data.regression_line.x[1], y: visualization_data.regression_line.y[1] },
      ]
    : null;

  // Colors for each set
  const colors = {
    train: '#3b82f6',     // blue
    validation: '#f59e0b', // amber
    test: '#10b981',       // green
  };

  return (
    <div className="space-y-6">
      {/* Metrics Comparison Cards */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Performance Metrics by Dataset Split
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Training Metrics */}
          <div className="border-2 border-blue-200 rounded-lg p-4 bg-blue-50">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="font-semibold text-blue-800">Training Set</span>
              <span className="text-xs text-blue-600">({split_info.train_size} samples)</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-blue-600">R²:</span>
                <span className="font-bold ml-1">{metrics.train.r2}</span>
              </div>
              <div>
                <span className="text-blue-600">MSE:</span>
                <span className="font-bold ml-1">{metrics.train.mse}</span>
              </div>
              <div>
                <span className="text-blue-600">RMSE:</span>
                <span className="font-bold ml-1">{metrics.train.rmse}</span>
              </div>
              <div>
                <span className="text-blue-600">MAE:</span>
                <span className="font-bold ml-1">{metrics.train.mae}</span>
              </div>
            </div>
          </div>

          {/* Validation Metrics */}
          <div className="border-2 border-amber-200 rounded-lg p-4 bg-amber-50">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 rounded-full bg-amber-500"></div>
              <span className="font-semibold text-amber-800">Validation Set</span>
              <span className="text-xs text-amber-600">({split_info.val_size} samples)</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-amber-600">R²:</span>
                <span className="font-bold ml-1">{metrics.validation.r2}</span>
              </div>
              <div>
                <span className="text-amber-600">MSE:</span>
                <span className="font-bold ml-1">{metrics.validation.mse}</span>
              </div>
              <div>
                <span className="text-amber-600">RMSE:</span>
                <span className="font-bold ml-1">{metrics.validation.rmse}</span>
              </div>
              <div>
                <span className="text-amber-600">MAE:</span>
                <span className="font-bold ml-1">{metrics.validation.mae}</span>
              </div>
            </div>
          </div>

          {/* Test Metrics */}
          <div className="border-2 border-green-200 rounded-lg p-4 bg-green-50">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="font-semibold text-green-800">Test Set</span>
              <span className="text-xs text-green-600">({split_info.test_size} samples)</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <span className="text-green-600">R²:</span>
                <span className="font-bold ml-1">{metrics.test.r2}</span>
              </div>
              <div>
                <span className="text-green-600">MSE:</span>
                <span className="font-bold ml-1">{metrics.test.mse}</span>
              </div>
              <div>
                <span className="text-green-600">RMSE:</span>
                <span className="font-bold ml-1">{metrics.test.rmse}</span>
              </div>
              <div>
                <span className="text-green-600">MAE:</span>
                <span className="font-bold ml-1">{metrics.test.mae}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Overfitting Analysis */}
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-medium text-gray-700 mb-2">Generalization Analysis</h4>
          <div className="text-sm text-gray-600">
            <p>
              <strong>Train → Validation gap:</strong>{' '}
              {(metrics.train.r2 - metrics.validation.r2).toFixed(4)} R² drop
              {metrics.train.r2 - metrics.validation.r2 > 0.1 && (
                <span className="text-red-600 ml-2">(potential overfitting)</span>
              )}
              {metrics.train.r2 - metrics.validation.r2 <= 0.05 && (
                <span className="text-green-600 ml-2">(good generalization)</span>
              )}
            </p>
            <p>
              <strong>Train → Test gap:</strong>{' '}
              {(metrics.train.r2 - metrics.test.r2).toFixed(4)} R² drop
            </p>
          </div>
        </div>
      </div>

      {/* Coefficients */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Final Coefficients
        </h3>
        <div className="flex flex-wrap gap-4">
          {coefficients.names.map((name, idx) => (
            <div
              key={name}
              className="bg-blue-50 border border-blue-200 rounded-lg px-4 py-3"
            >
              <div className="text-sm text-blue-600 font-medium">{name}</div>
              <div className="text-xl font-bold text-blue-800">
                {coefficients.values[idx]}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Scatter Plot with Regression Line - Shows all sets */}
      {featureColumns.length === 1 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Data with Fitted Line (All Sets)
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            Blue = Training, Amber = Validation, Green = Test. The red line is fit only on training data.
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="x"
                  type="number"
                  name={featureColumns[0]}
                  label={{ value: featureColumns[0], position: 'bottom' }}
                  stroke="#6b7280"
                  domain={['dataMin', 'dataMax']}
                />
                <YAxis
                  type="number"
                  name={targetColumn}
                  label={{ value: targetColumn, angle: -90, position: 'left' }}
                  stroke="#6b7280"
                />
                <Tooltip
                  formatter={(value, name) => [typeof value === 'number' ? value.toFixed(4) : value, name]}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                  }}
                />
                <Legend />
                <Scatter
                  name="Training"
                  data={trainData}
                  dataKey="actual"
                  fill={colors.train}
                  fillOpacity={0.7}
                />
                <Scatter
                  name="Validation"
                  data={valData}
                  dataKey="actual"
                  fill={colors.validation}
                  fillOpacity={0.7}
                />
                <Scatter
                  name="Test"
                  data={testData}
                  dataKey="actual"
                  fill={colors.test}
                  fillOpacity={0.7}
                />
                {lineData && (
                  <Line
                    data={lineData}
                    type="linear"
                    dataKey="y"
                    stroke="#ef4444"
                    strokeWidth={2}
                    dot={false}
                    name="Fitted Line"
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Residual Plot - All Sets */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Residual Plot (All Sets)
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Residuals should be randomly scattered around zero for all sets.
          Patterns may indicate model issues.
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="predicted"
                type="number"
                name="Predicted"
                label={{ value: 'Predicted Values', position: 'bottom' }}
                stroke="#6b7280"
              />
              <YAxis
                dataKey="residual"
                type="number"
                name="Residual"
                label={{ value: 'Residuals', angle: -90, position: 'left' }}
                stroke="#6b7280"
              />
              <Tooltip
                formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="5 5" />
              <Scatter
                name="Training"
                data={trainData}
                fill={colors.train}
                fillOpacity={0.7}
              />
              <Scatter
                name="Validation"
                data={valData}
                fill={colors.validation}
                fillOpacity={0.7}
              />
              <Scatter
                name="Test"
                data={testData}
                fill={colors.test}
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Actual vs Predicted - All Sets */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Actual vs Predicted (All Sets)
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Points closer to the diagonal indicate better predictions.
          Compare how well each set aligns with the perfect prediction line.
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="actual"
                type="number"
                name="Actual"
                label={{ value: 'Actual Values', position: 'bottom' }}
                stroke="#6b7280"
              />
              <YAxis
                dataKey="predicted"
                type="number"
                name="Predicted"
                label={{ value: 'Predicted Values', angle: -90, position: 'left' }}
                stroke="#6b7280"
              />
              <Tooltip
                formatter={(value) => typeof value === 'number' ? value.toFixed(4) : value}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Scatter
                name="Training"
                data={trainData}
                fill={colors.train}
                fillOpacity={0.7}
              />
              <Scatter
                name="Validation"
                data={valData}
                fill={colors.validation}
                fillOpacity={0.7}
              />
              <Scatter
                name="Test"
                data={testData}
                fill={colors.test}
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
