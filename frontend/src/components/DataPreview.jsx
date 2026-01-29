export default function DataPreview({ data }) {
  const { columns, preview, row_count, column_stats, filename } = data;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-800">Data Preview</h2>
        <div className="text-sm text-gray-500">
          {filename} - {row_count} rows, {columns.length} columns
        </div>
      </div>

      {/* Data Table */}
      <div className="overflow-x-auto mb-6">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {columns.map((col) => (
                <th
                  key={col}
                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                >
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {preview.map((row, idx) => (
              <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                {columns.map((col) => (
                  <td key={col} className="px-4 py-2 text-sm text-gray-700 whitespace-nowrap">
                    {row[col] !== null ? String(row[col]) : '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Column Statistics */}
      <h3 className="text-lg font-medium text-gray-700 mb-3">Column Statistics</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {columns.map((col) => {
          const stats = column_stats[col];
          return (
            <div key={col} className="stat-card">
              <div className="font-medium text-gray-800 mb-2">{col}</div>
              <div className="text-xs text-gray-500 mb-2">
                Type: {stats.type}
              </div>
              {stats.type === 'numeric' ? (
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    <span className="text-gray-500">Mean:</span> {stats.mean}
                  </div>
                  <div>
                    <span className="text-gray-500">Std:</span> {stats.std}
                  </div>
                  <div>
                    <span className="text-gray-500">Min:</span> {stats.min}
                  </div>
                  <div>
                    <span className="text-gray-500">Max:</span> {stats.max}
                  </div>
                </div>
              ) : (
                <div className="text-sm">
                  <span className="text-gray-500">Unique values:</span> {stats.unique}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
