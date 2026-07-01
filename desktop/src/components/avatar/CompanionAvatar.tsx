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
  // The L-shapes slide apart through the running/execution states, and the whole
  // logo spins while processing/executing.
  const animateLogo = !shouldReduceMotion && state !== 'idle';
  const spinLogo = !shouldReduceMotion && (state === 'processing' || state === 'executing');
  const [hover, setHover] = useState(false);

  const floatClass = shouldReduceMotion
    ? ''
    : state === 'confirming'
      ? 'avatar-attn'
      : state === 'follow_up'
        ? 'avatar-shimmer'
        : 'avatar-float';

  // glass-card box-shadow stack. Dark tinted glass: a soft inner black vignette
  // (instead of the source's near-solid white glow) plus a state-colored outer glow
  // so the avatar still expresses its state.
  const boxShadow = [
    '0 8px 32px rgba(0,0,0,0.10)',
    'inset 0 1px 0 rgba(255,255,255,0.28)',
    'inset 0 -1px 0 rgba(255,255,255,0.06)',
    'inset 0 0 55px 20px rgba(0,0,0,0.45)',
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
          background: 'rgba(6,8,14,0.55)',
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
        {/* companion double-L logo — white when idle, tints with state. The two
            conjugate L-shapes slide apart and back while active. The 13deg tilt
            is baked into the path coords. */}
        <div className="absolute inset-0 grid place-items-center">
          <svg
            width="70"
            height="70"
            viewBox="0 0 100 100"
            fill="none"
            aria-hidden="true"
            className={spinLogo ? 'companion-logo-spin' : ''}
            style={{ filter: `drop-shadow(0 0 8px ${logoColor}aa)`, overflow: 'visible' }}
          >
            <path
              className={`companion-logo-l ${animateLogo ? 'companion-logo-l-a' : ''}`}
              d="M30.4 10.6 L84.9 23.2 L75.5 64.1 L57.9 60 L63.8 34.7 L26.8 26.2 Z"
              fill={logoColor}
              style={{ transition: 'fill 0.35s ease' }}
            />
            <path
              className={`companion-logo-l ${animateLogo ? 'companion-logo-l-b' : ''}`}
              d="M69.6 89.4 L15.1 76.8 L24.5 35.9 L42.1 40 L36.2 65.3 L73.2 73.8 Z"
              fill={logoColor}
              style={{ transition: 'fill 0.35s ease' }}
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
