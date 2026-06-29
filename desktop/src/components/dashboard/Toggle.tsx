interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

export function Toggle({ label, checked, onChange, disabled = false }: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className="flex min-h-11 w-full items-center justify-between gap-4 rounded border border-white/10 bg-white/[0.04] px-3 py-2 text-left transition hover:bg-white/[0.07] disabled:cursor-not-allowed disabled:opacity-45"
    >
      <span className="text-sm font-medium text-white/82">{label}</span>
      <span
        className={`relative h-6 w-11 shrink-0 rounded-full border transition ${
          checked ? 'border-[#8EEBFF]/55 bg-[#8EEBFF]/28' : 'border-white/14 bg-white/[0.06]'
        }`}
      >
        <span
          className={`absolute top-1 h-4 w-4 rounded-full bg-white shadow transition ${
            checked ? 'left-6' : 'left-1'
          }`}
        />
      </span>
    </button>
  );
}
