import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import type { DialogueState, UICommand } from '../../protocol';
import { useJarvisStore } from '../../stores/jarvisStore';
import { openDashboard, showOverlay, hideOverlay, isTauri } from '../../lib/app';
import { Avatar } from '../Avatar';
import { DirectionPicker } from '../DirectionPicker';
import { Transcript } from '../Transcript';
import { PromptInput } from './PromptInput';

interface OverlayProps {
  send: (cmd: UICommand) => void;
}

// Time the dismiss animation needs before the native window is actually hidden.
const DISMISS_MS = 460;

export function Overlay({ send }: OverlayProps) {
  const muted = useJarvisStore((state) => state.muted);
  const setMuted = useJarvisStore((state) => state.setMuted);
  const dialogueState = useJarvisStore((state) => state.dialogueState);
  const prevState = useRef<DialogueState>('idle');

  // Wake-driven window lifecycle (Tauri): the overlay window stays hidden in the
  // background and pops up when the assistant becomes active, then sinks away and
  // hides when the interaction returns to idle.
  useEffect(() => {
    const prev = prevState.current;
    prevState.current = dialogueState;
    if (!isTauri()) return;
    if (dialogueState !== 'idle') {
      void showOverlay();
    } else if (prev !== 'idle') {
      const timer = window.setTimeout(() => {
        void hideOverlay();
      }, DISMISS_MS);
      return () => window.clearTimeout(timer);
    }
  }, [dialogueState]);

  const toggleMuted = () => {
    const nextMuted = !muted;
    setMuted(nextMuted);
    send({ type: 'mute_toggle', muted: nextMuted });
  };

  // Present (risen) while active. In the browser there is no native window to
  // show/hide, so keep it visible for development.
  const present = isTauri() ? dialogueState !== 'idle' : true;
  const stageStyle: CSSProperties = {
    transform: present ? 'translateY(0)' : 'translateY(115%)',
    opacity: present ? 1 : 0,
    transition: 'transform 400ms cubic-bezier(0.16, 1, 0.3, 1), opacity 320ms ease',
    willChange: 'transform, opacity',
  };

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-transparent text-white">
      <div className="absolute inset-0 flex flex-col" style={stageStyle}>
        <main className="flex flex-1 flex-col items-center justify-center gap-5 px-4">
          <Avatar />
          <Transcript />
        </main>

        <div className="m-3 flex items-center gap-2 rounded border border-white/10 bg-[#0A0A0F]/70 p-2 shadow-2xl shadow-black/40 backdrop-blur-xl">
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
            onClick={() => {
              void openDashboard();
            }}
            className="h-10 shrink-0 rounded border border-white/10 bg-white/[0.06] px-3 text-sm font-medium text-white/80 transition hover:border-[#8EEBFF]/35 hover:text-white"
          >
            Dashboard
          </button>
        </div>
      </div>

      {import.meta.env.DEV ? <DirectionPicker /> : null}
    </div>
  );
}
