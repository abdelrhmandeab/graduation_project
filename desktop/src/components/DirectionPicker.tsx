import type { AvatarDirection } from '../stores/jarvisStore';
import { useJarvisStore } from '../stores/jarvisStore';
import type { DialogueState } from '../protocol';

const directions: Array<{ id: AvatarDirection; label: string }> = [
  { id: 'aurora', label: 'Aurora' },
  { id: 'glyph', label: 'Glyph' },
  { id: 'glassai', label: 'Glass AI' },
  { id: 'companion', label: 'Companion' },
];

const states: DialogueState[] = ['idle', 'listening', 'processing', 'responding', 'confirming', 'executing', 'follow_up'];

export function DirectionPicker() {
  const avatarDirection = useJarvisStore((state) => state.avatarDirection);
  const previewDialogueState = useJarvisStore((state) => state.previewDialogueState);
  const setAvatarDirection = useJarvisStore((state) => state.setAvatarDirection);
  const previewState = useJarvisStore((state) => state.previewState);

  if (!import.meta.env.DEV) return null;

  return (
    <aside className="fixed bottom-4 right-4 z-50 w-[252px] rounded-md border border-white/10 bg-black/70 p-3 text-[11px] text-white shadow-2xl backdrop-blur">
      <div className="mb-2 text-xs font-semibold text-white/80">Avatar</div>
      <div className="mb-3 grid grid-cols-2 gap-1.5">
        {directions.map((direction) => (
          <button
            key={direction.id}
            type="button"
            onClick={() => setAvatarDirection(direction.id)}
            className={`rounded border px-2 py-1 transition-opacity hover:opacity-90 ${avatarDirection === direction.id ? 'border-cyan-200 bg-cyan-200/18 text-cyan-50' : 'border-white/10 bg-white/5 text-white/70'}`}
          >
            {direction.label}
          </button>
        ))}
      </div>
      <div className="mb-2 text-xs font-semibold text-white/80">State</div>
      <div className="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={() => previewState(null)}
          className={`rounded border px-2 py-1 ${previewDialogueState === null ? 'border-cyan-200 bg-cyan-200/18 text-cyan-50' : 'border-white/10 bg-white/5 text-white/70'}`}
        >
          Live
        </button>
        {states.map((state) => (
          <button
            key={state}
            type="button"
            onClick={() => previewState(state)}
            className={`rounded border px-2 py-1 capitalize ${previewDialogueState === state ? 'border-cyan-200 bg-cyan-200/18 text-cyan-50' : 'border-white/10 bg-white/5 text-white/70'}`}
          >
            {state.replace('_', ' ')}
          </button>
        ))}
      </div>
    </aside>
  );
}
