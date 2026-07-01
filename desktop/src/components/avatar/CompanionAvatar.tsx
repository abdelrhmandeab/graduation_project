import { useState, type CSSProperties } from 'react';
import { useReducedMotion } from 'motion/react';
import type { DialogueState } from '../../protocol';

interface AvatarProps {
  state: DialogueState;
  color: string;
}

export function CompanionAvatar({ state, color }: AvatarProps) {
  const active = state === 'idle' ? '#AEEFFF' : color;
  // Logo: white at rest, tinted with the state color once active.
  const logoColor = state === 'idle' ? '#FFFFFF' : color;
  const shouldReduceMotion = useReducedMotion();
  const [hover, setHover] = useState(false);

  const floatClass = shouldReduceMotion
    ? ''
    : state === 'confirming'
      ? 'avatar-attn'
      : state === 'follow_up'
        ? 'avatar-shimmer'
        : 'avatar-float';

  // glass-card box-shadow stack. The source's `inset 0 0 50px 25px rgba(255,255,255,2.5)`
  // had an invalid alpha (>1, clamps to 1 = near-solid white); tamed to 0.22, and a
  // state-colored outer glow added so the avatar still expresses its state.
  const boxShadow = [
    '0 8px 32px rgba(0,0,0,0.10)',
    'inset 0 1px 0 rgba(255,255,255,0.5)',
    'inset 0 -1px 0 rgba(255,255,255,0.1)',
    'inset 0 0 50px 25px rgba(255,255,255,0.22)',
    `0 0 26px ${active}55`,
    `0 0 60px ${active}22`,
  ].join(', ');

  return (
    <div
      className={`avatar-root state-${state} relative grid h-[220px] w-[220px] place-items-center`}
      style={{ '--state-color': color, '--bot-active': active } as CSSProperties}
    >
      <div
        className={floatClass}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        style={{
          position: 'relative',
          width: 142,
          height: 142,
          background: 'rgba(255,255,255,0.06)',
          backdropFilter: 'blur(5px)',
          WebkitBackdropFilter: 'blur(5px)',
          borderRadius: 20,
          border: `1.5px solid ${active}`,
          boxShadow,
          overflow: 'hidden',
          transform: hover && !shouldReduceMotion ? 'scale(1.05)' : 'scale(1)',
          transition: 'transform 0.8s cubic-bezier(0.175, 0.885, 0.32, 1.6)',
          willChange: 'transform',
        }}
      >
        {/* ::before — top edge highlight */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: 1,
            background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent)',
          }}
        />
        {/* ::after — left edge highlight */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: 1,
            height: '100%',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.8), transparent, rgba(255,255,255,0.3))',
          }}
        />
        {/* state-colored core glow seen through the frosted glass */}
        <div
          className="avatar-core-pulse"
          style={{
            position: 'absolute',
            inset: 0,
            background: `radial-gradient(circle at 50% 46%, ${active}55 0%, ${active}1a 44%, transparent 72%)`,
          }}
        />
        {/* companion double-L logo — white when idle, tints with state,
            shrinks/expands through the running + execution states */}
        <div className="absolute inset-0 grid place-items-center">
          <svg
            width="70"
            height="70"
            viewBox="0 0 100 100"
            fill="none"
            aria-hidden="true"
            className={!shouldReduceMotion && state !== 'idle' ? 'companion-logo-throb' : ''}
            style={{ filter: `drop-shadow(0 0 8px ${logoColor}aa)` }}
          >
            <path
              d="M22 16 L78 16 L78 58 L60 58 L60 32 L22 32 Z M78 84 L22 84 L22 42 L40 42 L40 68 L78 68 Z"
              transform="rotate(13 50 50)"
              fill={logoColor}
              style={{ transition: 'fill 0.35s ease' }}
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
