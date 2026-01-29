import { useState } from 'react';
import MathStep from './MathStep';

export default function StepByStep({ steps }) {
  const [expandedSteps, setExpandedSteps] = useState(
    steps.map((_, idx) => idx === 0) // First step expanded by default
  );
  const [currentStep, setCurrentStep] = useState(0);

  const toggleStep = (idx) => {
    const newExpanded = [...expandedSteps];
    newExpanded[idx] = !newExpanded[idx];
    setExpandedSteps(newExpanded);
  };

  const expandAll = () => {
    setExpandedSteps(steps.map(() => true));
  };

  const collapseAll = () => {
    setExpandedSteps(steps.map(() => false));
  };

  const goToStep = (idx) => {
    setCurrentStep(idx);
    const newExpanded = steps.map((_, i) => i === idx);
    setExpandedSteps(newExpanded);
  };

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      goToStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      goToStep(currentStep - 1);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">
          Step-by-Step Explanation
        </h2>
        <div className="flex gap-2">
          <button
            onClick={collapseAll}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
          >
            Collapse All
          </button>
          <button
            onClick={expandAll}
            className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800"
          >
            Expand All
          </button>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-gray-500">
            Step {currentStep + 1} of {steps.length}
          </span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-600 transition-all duration-300"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {/* Step Navigation Buttons */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={prevStep}
          disabled={currentStep === 0}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
            currentStep === 0
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Previous Step
        </button>
        <button
          onClick={nextStep}
          disabled={currentStep === steps.length - 1}
          className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
            currentStep === steps.length - 1
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700'
          }`}
        >
          Next Step
        </button>
      </div>

      {/* Steps */}
      <div className="space-y-4">
        {steps.map((step, idx) => (
          <MathStep
            key={idx}
            step={step}
            isExpanded={expandedSteps[idx]}
            onToggle={() => toggleStep(idx)}
          />
        ))}
      </div>
    </div>
  );
}
