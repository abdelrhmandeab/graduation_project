import { useEffect, useRef } from 'react';
import type { CSSProperties } from 'react';
import type { DialogueState, UICommand } from '../../protocol';
import { STATE_COLORS } from '../../protocol';
import { useJarvisStore } from '../../stores/jarvisStore';
import { openDashboard, showOverlay, hideOverlay, isTauri } from '../../lib/app';
import { Avatar } from '../Avatar';
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
  const textPromptEnabled = useJarvisStore((state) => state.textPromptEnabled);
  const stateColor = STATE_COLORS[dialogueState];
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
        {/* persistent ambient light at the bottom, tinted with the state color —
            stays visible even when the input bar is hidden */}
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-40"
          style={{
            backgroundColor: stateColor,
            opacity: 0.4,
            filter: 'blur(48px)',
            // radial mask so the glow fades softly to transparent at every edge
            // instead of getting clipped square by the window's overflow.
            WebkitMaskImage: 'radial-gradient(72% 85% at 50% 100%, #000 0%, transparent 72%)',
            maskImage: 'radial-gradient(72% 85% at 50% 100%, #000 0%, transparent 72%)',
            transition: 'background-color 0.4s ease',
          }}
        />
        <main className="flex flex-1 flex-col items-center justify-center gap-5 px-4">
          <Avatar />
          <Transcript />
        </main>

        {textPromptEnabled ? (
          <div className="relative m-3 flex items-center gap-2 rounded border border-white/10 bg-[#0A0A0F]/70 p-2 shadow-2xl shadow-black/40 backdrop-blur-xl">
            <PromptInput send={send} />
          <button
            type="button"
            onClick={toggleMuted}
            aria-label={muted ? 'Unmute' : 'Mute'}
            title={muted ? 'Unmute' : 'Mute'}
            className={`grid h-10 w-10 shrink-0 place-items-center rounded border transition ${
              muted
                ? 'border-red-300/30 bg-red-400/15 text-red-100 hover:bg-red-400/20'
                : 'border-white/10 bg-white/[0.06] text-white/75 hover:bg-white/[0.1]'
            }`}
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M4 9 L8 9 L13 5 L13 19 L8 15 L4 15 Z"
                fill="currentColor"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinejoin="round"
              />
              {muted ? (
                <path d="M17 9 L22 15 M22 9 L17 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              ) : (
                <path
                  d="M16.5 8.5 A5 5 0 0 1 16.5 15.5 M19 6 A9 9 0 0 1 19 18"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                />
              )}
            </svg>
          </button>
          <button
            type="button"
            onClick={() => {
              void openDashboard();
            }}
            aria-label="Dashboard"
            title="Dashboard"
            className="grid h-10 w-10 shrink-0 place-items-center rounded border border-white/10 bg-white/[0.06] text-white/80 transition hover:border-[#8EEBFF]/35 hover:text-white"
          >
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M4 4 H10 V11 H4 Z M14 4 H20 V8 H14 Z M14 12 H20 V20 H14 Z M4 15 H10 V20 H4 Z"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
