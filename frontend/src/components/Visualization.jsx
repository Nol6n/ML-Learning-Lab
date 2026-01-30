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
  const { visualization_data, metrics, coefficients, coefficient_table, sklearn_verification } = result;

  // Prepare scatter plot data
  const scatterData = visualization_data.actual.map((actual, idx) => ({
    x: visualization_data.feature_values[idx],
    actual,
    predicted: visualization_data.predicted[idx],
    residual: visualization_data.residuals[idx],
  }));

  // Regression line data (for single feature)
  const lineData = visualization_data.regression_line
    ? [
        { x: visualization_data.regression_line.x[0], y: visualization_data.regression_line.y[0] },
        { x: visualization_data.regression_line.x[1], y: visualization_data.regression_line.y[1] },
      ]
    : null;

  return (
    <div className="space-y-6">
      {/* Model Summary Card */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Model Summary
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <div className="text-blue-600 text-xs uppercase font-medium">R² (Coefficient of Determination)</div>
            <div className="text-2xl font-bold text-blue-800">{metrics.r2}</div>
            <div className="text-sm text-blue-600 mt-1">
              {(metrics.r2 * 100).toFixed(1)}% of variance explained
            </div>
          </div>
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="text-green-600 text-xs uppercase font-medium">RSE (Residual Std Error)</div>
            <div className="text-2xl font-bold text-green-800">{metrics.rse}</div>
            <div className="text-sm text-green-600 mt-1">
              typical prediction error
            </div>
          </div>
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <div className="text-purple-600 text-xs uppercase font-medium">F-statistic</div>
            <div className="text-2xl font-bold text-purple-800">{metrics.f_statistic}</div>
            <div className="text-sm text-purple-600 mt-1">
              p-value: {metrics.f_pvalue < 0.0001 ? '< 0.0001' : metrics.f_pvalue.toFixed(4)}
            </div>
          </div>
          <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
            <div className="text-amber-600 text-xs uppercase font-medium">Sample Size</div>
            <div className="text-2xl font-bold text-amber-800">n = {result.sample_size}</div>
            <div className="text-sm text-amber-600 mt-1">
              df = {result.degrees_of_freedom}
            </div>
          </div>
        </div>
      </div>

      {/* Coefficient Table */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Coefficient Estimates (with Standard Errors & Tests)
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Small p-values (typically &lt; 0.05) indicate statistically significant predictors.
        </p>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-700"></th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Coefficient</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">Std. Error</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">t-statistic</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">p-value</th>
                <th className="text-right py-3 px-4 font-semibold text-gray-700">95% CI</th>
              </tr>
            </thead>
            <tbody>
              {coefficient_table.rows.map((row, idx) => (
                <tr
                  key={idx}
                  className={`border-b border-gray-100 ${
                    row.p === '< 0.0001' || parseFloat(row.p) < 0.05
                      ? 'bg-green-50'
                      : ''
                  }`}
                >
                  <td className="py-3 px-4 font-medium text-gray-800">{row.name}</td>
                  <td className="text-right py-3 px-4 font-mono">{row.coef}</td>
                  <td className="text-right py-3 px-4 font-mono text-gray-600">{row.se}</td>
                  <td className="text-right py-3 px-4 font-mono">{row.t}</td>
                  <td className={`text-right py-3 px-4 font-mono ${
                    row.p === '< 0.0001' || parseFloat(row.p) < 0.05
                      ? 'text-green-700 font-semibold'
                      : 'text-gray-600'
                  }`}>
                    {row.p}
                    {(row.p === '< 0.0001' || parseFloat(row.p) < 0.05) && ' ***'}
                  </td>
                  <td className="text-right py-3 px-4 font-mono text-gray-600">{row.ci}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-3">
          *** indicates statistical significance at the 0.05 level. Green rows show significant predictors.
        </p>
      </div>

      {/* sklearn Verification Card */}
      {sklearn_verification && (
        <div className={`rounded-lg shadow-sm border p-6 ${
          sklearn_verification.all_match
            ? 'bg-green-50 border-green-200'
            : 'bg-red-50 border-red-200'
        }`}>
          <h3 className={`text-lg font-semibold mb-2 ${
            sklearn_verification.all_match ? 'text-green-800' : 'text-red-800'
          }`}>
            {sklearn_verification.all_match ? '✓ sklearn Verification Passed!' : '✗ sklearn Verification Failed'}
          </h3>
          <p className={`text-sm ${
            sklearn_verification.all_match ? 'text-green-700' : 'text-red-700'
          }`}>
            {sklearn_verification.all_match
              ? 'Our hand-calculated coefficients match sklearn\'s LinearRegression exactly. The math is correct!'
              : 'There\'s a discrepancy between our calculations and sklearn. Double-check the implementation.'}
          </p>
          <div className="mt-3 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className={sklearn_verification.all_match ? 'text-green-600' : 'text-red-600'}>
                sklearn R²:
              </span>
              <span className="font-mono ml-2">{sklearn_verification.r2.toFixed(6)}</span>
            </div>
            <div>
              <span className={sklearn_verification.all_match ? 'text-green-600' : 'text-red-600'}>
                Our R²:
              </span>
              <span className="font-mono ml-2">{metrics.r2.toFixed ? metrics.r2.toFixed(6) : metrics.r2}</span>
            </div>
          </div>
        </div>
      )}

      {/* Scatter Plot with Regression Line */}
      {featureColumns.length === 1 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">
            Data with Fitted Regression Line
          </h3>
          <p className="text-sm text-gray-600 mb-4">
            The red line is our least squares fit: <span className="font-mono">
              {targetColumn} = {coefficients.values[0].toFixed(2)} + {coefficients.values[1].toFixed(4)} × {featureColumns[0]}
            </span>
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="x"
                  type="number"
                  name={featureColumns[0]}
                  label={{ value: featureColumns[0], position: 'bottom', offset: 0 }}
                  stroke="#6b7280"
                  domain={['dataMin', 'dataMax']}
                />
                <YAxis
                  type="number"
                  name={targetColumn}
                  label={{ value: targetColumn, angle: -90, position: 'insideLeft' }}
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
                  name="Data Points"
                  data={scatterData}
                  dataKey="actual"
                  fill="#3b82f6"
                  fillOpacity={0.7}
                />
                {lineData && (
                  <Line
                    data={lineData}
                    type="linear"
                    dataKey="y"
                    stroke="#ef4444"
                    strokeWidth={3}
                    dot={false}
                    name="Fitted Line"
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Residual Plot */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Residual Plot
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Residuals should be randomly scattered around zero with no clear pattern.
          Patterns might indicate non-linearity, heteroscedasticity, or other issues!
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="predicted"
                type="number"
                name="Predicted"
                label={{ value: 'Fitted Values (ŷ)', position: 'bottom', offset: 0 }}
                stroke="#6b7280"
              />
              <YAxis
                dataKey="residual"
                type="number"
                name="Residual"
                label={{ value: 'Residuals (e)', angle: -90, position: 'insideLeft' }}
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
              <ReferenceLine y={0} stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" />
              <Scatter
                name="Residuals"
                data={scatterData}
                fill="#3b82f6"
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Actual vs Predicted */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Actual vs Predicted
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Points close to the diagonal line indicate good predictions.
          Perfect predictions would lie exactly on the dashed line.
        </p>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 40, left: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis
                dataKey="actual"
                type="number"
                name="Actual"
                label={{ value: 'Actual Values', position: 'bottom', offset: 0 }}
                stroke="#6b7280"
              />
              <YAxis
                dataKey="predicted"
                type="number"
                name="Predicted"
                label={{ value: 'Predicted Values', angle: -90, position: 'insideLeft' }}
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
              {/* Perfect prediction line - would need to calculate based on data range */}
              <Scatter
                name="Predictions"
                data={scatterData}
                fill="#10b981"
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Final Coefficients Summary */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Final Model Equation
        </h3>
        <div className="bg-gray-50 rounded-lg p-4 font-mono text-center text-lg">
          {featureColumns.length === 1 ? (
            <span>
              <span className="text-blue-600">{targetColumn}</span> = {' '}
              <span className="text-purple-600">{coefficients.values[0].toFixed(4)}</span> + {' '}
              <span className="text-green-600">{coefficients.values[1].toFixed(4)}</span> × {' '}
              <span className="text-blue-600">{featureColumns[0]}</span>
            </span>
          ) : (
            <span>
              <span className="text-blue-600">{targetColumn}</span> = {' '}
              <span className="text-purple-600">{coefficients.values[0].toFixed(4)}</span>
              {featureColumns.map((col, idx) => (
                <span key={col}>
                  {' '} + <span className="text-green-600">{coefficients.values[idx + 1].toFixed(4)}</span> × {' '}
                  <span className="text-blue-600">{col}</span>
                </span>
              ))}
            </span>
          )}
        </div>
        <div className="mt-4 flex flex-wrap gap-3">
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
    </div>
  );
}
