import { useEffect, useRef, useState } from 'react';
import katex from 'katex';

function RenderLatex({ latex }) {
  const containerRef = useRef(null);

  useEffect(() => {
    if (containerRef.current && latex) {
      try {
        katex.render(latex, containerRef.current, {
          displayMode: true,
          throwOnError: false,
          trust: true,
        });
      } catch (e) {
        containerRef.current.textContent = latex;
      }
    }
  }, [latex]);

  if (!latex) return null;

  return <div ref={containerRef} className="math-content" />;
}

function StatisticsTable({ values }) {
  if (!values || typeof values !== 'object') return null;

  // Check if it's coefficient statistics table (from Step 6)
  if (values.coefficients && Array.isArray(values.coefficients)) {
    return (
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
          <thead className="bg-gray-50">
            <tr>
              <th className="text-left py-2 px-3 font-semibold text-gray-700 border-b"></th>
              <th className="text-right py-2 px-3 font-semibold text-gray-700 border-b">Coefficient</th>
              <th className="text-right py-2 px-3 font-semibold text-gray-700 border-b">Std. Error</th>
              <th className="text-right py-2 px-3 font-semibold text-gray-700 border-b">t-stat</th>
              <th className="text-right py-2 px-3 font-semibold text-gray-700 border-b">p-value</th>
              <th className="text-right py-2 px-3 font-semibold text-gray-700 border-b">95% CI</th>
            </tr>
          </thead>
          <tbody>
            {values.coefficients.map((row, idx) => {
              const isSignificant = row.p === '< 0.0001' || parseFloat(row.p) < 0.05;
              return (
                <tr key={idx} className={`${isSignificant ? 'bg-green-50' : 'bg-white'} border-b border-gray-100`}>
                  <td className="py-2 px-3 font-medium text-gray-800">{row.name}</td>
                  <td className="text-right py-2 px-3 font-mono">{row.coef}</td>
                  <td className="text-right py-2 px-3 font-mono text-gray-600">{row.se}</td>
                  <td className="text-right py-2 px-3 font-mono">{row.t}</td>
                  <td className={`text-right py-2 px-3 font-mono ${isSignificant ? 'text-green-700 font-semibold' : 'text-gray-600'}`}>
                    {row.p} {isSignificant && '***'}
                  </td>
                  <td className="text-right py-2 px-3 font-mono text-gray-500 text-xs">
                    [{row.ci_low}, {row.ci_high}]
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="text-xs text-gray-500 mt-2">*** p &lt; 0.05 (statistically significant)</p>
      </div>
    );
  }

  // Check if it's coefficient interpretations (array with name/value/interpretation)
  if (Array.isArray(values) && values[0]?.interpretation !== undefined) {
    return (
      <div className="mt-3 space-y-2">
        {values.map((item, idx) => (
          <div key={idx} className="bg-gray-50 p-3 rounded-lg">
            <div className="font-medium text-gray-800">
              {item.name} = {item.value}
            </div>
            <div className="text-sm text-gray-600 mt-1">
              {item.interpretation}
            </div>
          </div>
        ))}
      </div>
    );
  }

  // Check if it's a comparison table (array of set metrics)
  if (values.comparison && Array.isArray(values.comparison)) {
    const colors = {
      Training: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-800' },
      Validation: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-800' },
      Test: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800' },
    };
    return (
      <div className="mt-3 overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200">
              <th className="text-left py-2 px-3 font-medium text-gray-700">Set</th>
              <th className="text-right py-2 px-3 font-medium text-gray-700">R²</th>
              <th className="text-right py-2 px-3 font-medium text-gray-700">MSE</th>
              <th className="text-right py-2 px-3 font-medium text-gray-700">RMSE</th>
            </tr>
          </thead>
          <tbody>
            {values.comparison.map((row, idx) => {
              const style = colors[row.set] || { bg: 'bg-gray-50', border: 'border-gray-200', text: 'text-gray-800' };
              return (
                <tr key={idx} className={`${style.bg} ${style.border} border-l-4`}>
                  <td className={`py-2 px-3 font-medium ${style.text}`}>{row.set}</td>
                  <td className="text-right py-2 px-3">{row.r2}</td>
                  <td className="text-right py-2 px-3">{row.mse}</td>
                  <td className="text-right py-2 px-3">{row.rmse}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  // Check if it's single set metrics (has "set" field)
  if (values.set !== undefined && values.r2 !== undefined) {
    const colors = {
      Training: { bg: 'bg-blue-50', border: 'border-blue-200', label: 'text-blue-600' },
      Validation: { bg: 'bg-amber-50', border: 'border-amber-200', label: 'text-amber-600' },
      Test: { bg: 'bg-green-50', border: 'border-green-200', label: 'text-green-600' },
    };
    const style = colors[values.set] || { bg: 'bg-gray-50', border: 'border-gray-200', label: 'text-gray-600' };
    return (
      <div className={`grid grid-cols-4 gap-3 mt-3 p-3 rounded-lg ${style.bg} border ${style.border}`}>
        <div>
          <div className={`text-xs uppercase ${style.label}`}>R²</div>
          <div className="font-bold">{values.r2}</div>
        </div>
        <div>
          <div className={`text-xs uppercase ${style.label}`}>MSE</div>
          <div className="font-bold">{values.mse}</div>
        </div>
        <div>
          <div className={`text-xs uppercase ${style.label}`}>RMSE</div>
          <div className="font-bold">{values.rmse}</div>
        </div>
        <div>
          <div className={`text-xs uppercase ${style.label}`}>MAE</div>
          <div className="font-bold">{values.mae}</div>
        </div>
      </div>
    );
  }

  // Check if it's metrics without set field (old format)
  if (values.r2 !== undefined) {
    return (
      <div className="grid grid-cols-3 gap-4 mt-3">
        <div className="stat-card">
          <div className="stat-label">R² Score</div>
          <div className="stat-value">{values.r2}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">MSE</div>
          <div className="stat-value">{values.mse}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">RMSE</div>
          <div className="stat-value">{values.rmse}</div>
        </div>
      </div>
    );
  }

  // Feature statistics
  const entries = Object.entries(values);
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-3">
      {entries.map(([key, stats]) => (
        <div key={key} className="bg-gray-50 p-3 rounded-lg">
          <div className="font-medium text-gray-800 mb-2">{key}</div>
          {typeof stats === 'object' && stats.mean !== undefined && (
            <div className="grid grid-cols-2 gap-1 text-sm">
              <div><span className="text-gray-500">Mean:</span> {stats.mean}</div>
              <div><span className="text-gray-500">Std:</span> {stats.std}</div>
              <div><span className="text-gray-500">Min:</span> {stats.min}</div>
              <div><span className="text-gray-500">Max:</span> {stats.max}</div>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function DropdownSection({ title, description, subSteps }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="my-3 border border-blue-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-3 bg-blue-50 hover:bg-blue-100 transition-colors text-left"
      >
        <div>
          <span className="font-medium text-blue-800">{title}</span>
          {description && (
            <p className="text-sm text-blue-600 mt-1">{description}</p>
          )}
        </div>
        <svg
          className={`w-5 h-5 text-blue-600 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      {isOpen && (
        <div className="p-4 bg-white border-t border-blue-200">
          {subSteps.map((subStep, idx) => (
            <div key={idx} className="sub-step">
              <p className="text-gray-700 text-sm">{subStep.description}</p>
              {subStep.latex && <RenderLatex latex={subStep.latex} />}
              {subStep.note && (
                <p className="text-xs text-gray-500 mt-1 italic">{subStep.note}</p>
              )}
              {subStep.values && <StatisticsTable values={subStep.values} />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function MathStep({ step, isExpanded, onToggle }) {
  const { step_number, title, explanation, latex, sub_steps } = step;

  return (
    <div className="math-step">
      <div
        className="math-step-header cursor-pointer"
        onClick={onToggle}
      >
        <div className="math-step-number">{step_number}</div>
        <div className="flex-1">
          <h3 className="math-step-title">{title}</h3>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${
            isExpanded ? 'rotate-180' : ''
          }`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <p className="text-gray-700 leading-relaxed">{explanation}</p>

          {latex && <RenderLatex latex={latex} />}

          {sub_steps && sub_steps.length > 0 && (
            <div className="mt-4">
              {sub_steps.map((subStep, idx) => (
                subStep.type === 'dropdown' ? (
                  <DropdownSection
                    key={idx}
                    title={subStep.title}
                    description={subStep.description}
                    subSteps={subStep.sub_steps}
                  />
                ) : (
                  <div key={idx} className="sub-step">
                    <p className="text-gray-700 text-sm">{subStep.description}</p>
                    {subStep.latex && <RenderLatex latex={subStep.latex} />}
                    {subStep.note && (
                      <p className="text-xs text-gray-500 mt-1 italic">{subStep.note}</p>
                    )}
                    {subStep.values && <StatisticsTable values={subStep.values} />}
                  </div>
                )
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
