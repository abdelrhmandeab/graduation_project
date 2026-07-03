import type { ReactNode } from 'react';

/**
 * Floating glass panel — the translucent, blurred control surface from the
 * original avatar DirectionPicker. Pin it with position/size utilities via
 * `className` (e.g. "fixed bottom-4 right-4 w-[252px]").
 */
export function FloatingPanel({
  className = '',
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  return (
    <aside
      className={`z-50 rounded-md border border-black/10 bg-white/70 p-3 text-[11px] text-gray-900 shadow-2xl backdrop-blur dark:border-white/10 dark:bg-black/70 dark:text-white ${className}`}
    >
      {children}
    </aside>
  );
}

/** Small uppercase-ish section label used inside a FloatingPanel. */
export function PanelLabel({ children }: { children: ReactNode }) {
  return <div className="mb-2 text-xs font-semibold text-gray-600 dark:text-white/80">{children}</div>;
}

/**
 * Selectable pill toggle — cyan when active, faint white when not. The visual
 * building block of the avatar/state picker grid; reuse for any compact
 * single/multi-select chip group.
 */
export function Chip({
  active = false,
  onClick,
  className = '',
  children,
}: {
  active?: boolean;
  onClick?: () => void;
  className?: string;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded border px-2 py-1 transition-opacity hover:opacity-90 ${
        active
          ? 'border-cyan-400 bg-cyan-400/15 text-cyan-800 dark:border-cyan-200 dark:bg-cyan-200/18 dark:text-cyan-50'
          : 'border-black/10 bg-black/[0.04] text-gray-600 dark:border-white/10 dark:bg-white/5 dark:text-white/70'
      } ${className}`}
    >
      {children}
    </button>
  );
}
