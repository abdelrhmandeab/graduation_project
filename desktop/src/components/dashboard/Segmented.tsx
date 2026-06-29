export interface SegmentedOption<T extends string> {
  label: string;
  value: T;
}

interface SegmentedProps<T extends string> {
  value: T;
  options: Array<SegmentedOption<T>>;
  onChange: (value: T) => void;
  disabled?: boolean;
}

export function Segmented<T extends string>({ value, options, onChange, disabled = false }: SegmentedProps<T>) {
  return (
    <div className="grid gap-1 rounded border border-white/10 bg-black/25 p-1 sm:grid-flow-col sm:auto-cols-fr">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(option.value)}
          className={`min-h-9 rounded px-3 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-45 ${
            value === option.value
              ? 'bg-[#8EEBFF]/16 text-[#DDFBFF] shadow-[inset_0_0_0_1px_rgba(142,235,255,0.28)]'
              : 'text-white/62 hover:bg-white/[0.06] hover:text-white/85'
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
