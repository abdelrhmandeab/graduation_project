import type { CSSProperties } from 'react';
import { useReducedMotion } from 'motion/react';
import type { DialogueState } from '../../protocol';
import ShapeBlur from './ShapeBlur';

interface AvatarProps {
  state: DialogueState;
  color: string;
}

// Auto-orbit speed of the reveal point per state — animates without hover.
function autoSpeedForState(state: DialogueState): number {
  switch (state) {
    case 'processing':
      return 2.0;
    case 'responding':
      return 1.5;
    case 'listening':
    case 'confirming':
      return 1.2;
    case 'executing':
      return 0.8;
    case 'idle':
    case 'follow_up':
      return 0.5;
    default:
      return 0.9;
  }
}

export function GlassAIAvatar({ state, color }: AvatarProps) {
  const active = state === 'idle' ? '#AEEFFF' : color;
  const shouldReduceMotion = useReducedMotion();

  // Static fallback (centered rounded tile outline) if no WebGL context exists.
  const fallback = (
    <div
      className="absolute left-1/2 top-1/2 h-[150px] w-[150px] -translate-x-1/2 -translate-y-1/2 rounded-[34px]"
      style={{ border: `1.5px solid ${active}`, boxShadow: `0 0 22px ${active}66` }}
    />
  );

  return (
    <div
      className={`avatar-root state-${state} relative h-[220px] w-[220px]`}
      style={{ '--state-color': color, '--bot-active': active } as CSSProperties}
    >
      <ShapeBlur
        variation={0}
        color={active}
        shapeSize={1.2}
        roundness={0.4}
        borderSize={0.05}
        circleSize={0.3}
        circleEdge={0.5}
        pixelRatioProp={2}
        autoSpeed={autoSpeedForState(state)}
        pulseKey={state}
        animate={!shouldReduceMotion}
        fallback={fallback}
      />
    </div>
  );
}
