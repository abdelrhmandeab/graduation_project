import type { CSSProperties } from 'react';
import { useReducedMotion } from 'motion/react';
import type { DialogueState } from '../../protocol';
import Strands from './Strands';

interface AvatarProps {
  state: DialogueState;
  amplitude: number;
  color: string;
}

// State drives the strand motion: faster/brighter when the assistant is active.
function strandParamsForState(state: DialogueState, amplitude: number) {
  const amp = Math.min(1, Math.max(0, amplitude));
  switch (state) {
    case 'listening':
      return { speed: 0.6, amplitude: 0.9 + amp * 1.4, glow: 2.9, intensity: 0.72 };
    case 'processing':
      return { speed: 1.1, amplitude: 1.0, glow: 2.7, intensity: 0.7 };
    case 'responding':
      return { speed: 0.85, amplitude: 1.05 + amp * 0.8, glow: 3.0, intensity: 0.75 };
    case 'confirming':
      return { speed: 0.7, amplitude: 1.0, glow: 2.8, intensity: 0.74 };
    case 'executing':
      return { speed: 0.5, amplitude: 0.85, glow: 2.9, intensity: 0.7 };
    case 'follow_up':
      return { speed: 0.4, amplitude: 0.8, glow: 2.6, intensity: 0.62 };
    default: // idle — calm, slow drift
      return { speed: 0.28, amplitude: 0.7, glow: 2.4, intensity: 0.55 };
  }
}

export function AuroraAvatar({ state, amplitude, color }: AvatarProps) {
  const shouldReduceMotion = useReducedMotion();
  const params = strandParamsForState(state, amplitude);

  // Palette carries the current state color with a cold-glass cyan highlight.
  const colors = [color, '#8EEBFF', color, color];

  // Static CSS-orb fallback if no WebGL context is available.
  const fallback = (
    <div
      className="absolute left-1/2 top-1/2 h-[132px] w-[132px] -translate-x-1/2 -translate-y-1/2 rounded-full"
      style={{
        background: `radial-gradient(circle at 38% 32%, rgba(255,255,255,0.96) 0%, rgba(255,255,255,0.48) 14%, ${color} 48%, color-mix(in srgb, ${color} 42%, #05070D) 100%)`,
        boxShadow: `0 0 24px ${color}88, 0 0 58px ${color}33, inset -18px -24px 36px rgba(0,0,0,0.34), inset 12px 10px 18px rgba(255,255,255,0.18)`,
      }}
    />
  );

  return (
    <div
      className={`avatar-root state-${state} relative h-[220px] w-[220px]`}
      style={{ '--orb-color': color, '--state-color': color } as CSSProperties}
    >
      <Strands
        colors={colors}
        count={5}
        speed={params.speed}
        amplitude={params.amplitude}
        glow={params.glow}
        intensity={params.intensity}
        waviness={1.1}
        thickness={0.8}
        taper={3}
        saturation={1.4}
        scale={1.35}
        glass
        glassSize={1}
        refraction={1}
        dispersion={1}
        animate={!shouldReduceMotion}
        fallback={fallback}
      />
    </div>
  );
}
