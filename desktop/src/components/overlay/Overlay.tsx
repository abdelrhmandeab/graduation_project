import type { UICommand } from '../../protocol';
import { useJarvisStore } from '../../stores/jarvisStore';
import { Avatar } from '../Avatar';
import { DirectionPicker } from '../DirectionPicker';
import { Transcript } from '../Transcript';
import { PromptInput } from './PromptInput';

interface OverlayProps {
  send: (cmd: UICommand) => void;
}

export function Overlay({ send }: OverlayProps) {
  const muted = useJarvisStore((state) => state.muted);
  const setMuted = useJarvisStore((state) => state.setMuted);
  const setAppView = useJarvisStore((state) => state.setAppView);

  const toggleMuted = () => {
    const nextMuted = !muted;
    setMuted(nextMuted);
    send({ type: 'mute_toggle', muted: nextMuted });
  };

  return (
    <div className="min-h-screen bg-transparent text-white">
      <main className="flex min-h-screen flex-col items-center justify-center gap-6 px-4 pb-24">
        <Avatar />
        <Transcript />
      </main>

      <div className="fixed inset-x-0 bottom-5 z-40 mx-auto flex w-[min(720px,calc(100vw-32px))] items-center gap-2 rounded border border-white/10 bg-[#0A0A0F]/70 p-2 shadow-2xl shadow-black/40 backdrop-blur-xl">
        <PromptInput send={send} />
        <button
          type="button"
          onClick={toggleMuted}
          className={`h-10 shrink-0 rounded border px-3 text-sm font-medium transition ${
            muted
              ? 'border-red-300/30 bg-red-400/15 text-red-100 hover:bg-red-400/20'
              : 'border-white/10 bg-white/[0.06] text-white/75 hover:bg-white/[0.1]'
          }`}
        >
          {muted ? 'Muted' : 'Mute'}
        </button>
        <button
          type="button"
          onClick={() => setAppView('dashboard')}
          className="h-10 shrink-0 rounded border border-white/10 bg-white/[0.06] px-3 text-sm font-medium text-white/80 transition hover:border-[#8EEBFF]/35 hover:text-white"
        >
          Dashboard
        </button>
      </div>

      {import.meta.env.DEV ? <DirectionPicker /> : null}
    </div>
  );
}
