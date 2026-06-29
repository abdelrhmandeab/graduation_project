import { useEffect, useRef } from 'react';
import { useReducedMotion } from 'motion/react';
import type { DialogueState } from '../../protocol';

interface AvatarProps {
  state: DialogueState;
  amplitude: number;
  color: string;
}

const canvasStates = new Set<DialogueState>(['listening', 'responding']);

export function GlyphAvatar({ state, amplitude, color }: AvatarProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shouldReduceMotion = useReducedMotion();
  const canvasActive = canvasStates.has(state) && !shouldReduceMotion;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !canvasActive) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let tick = 0;
    let running = !document.hidden;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = 220 * dpr;
      canvas.height = 220 * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      ctx.clearRect(0, 0, 220, 220);
      tick += 0.045;
      const amp = Math.min(1, Math.max(0, amplitude));
      for (let i = 0; i < 28; i += 1) {
        const angle = (i / 28) * Math.PI * 2 - Math.PI / 2;
        const phase = Math.sin(tick * (state === 'responding' ? 2.2 : 1.4) + i * 0.55);
        const scale = state === 'responding' ? 0.65 + Math.max(0, phase) * 1.1 : 0.75 + (phase * 0.5 + 0.5) * (0.45 + amp);
        const radius = 84 + scale * 4;
        ctx.globalAlpha = 0.18 + Math.max(0, phase) * 0.72;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(110 + Math.cos(angle) * radius, 110 + Math.sin(angle) * radius, 2.2 * scale, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      if (running) raf = requestAnimationFrame(draw);
    };

    const onVisibility = () => {
      running = !document.hidden;
      if (running) {
        cancelAnimationFrame(raf);
        raf = requestAnimationFrame(draw);
      } else {
        cancelAnimationFrame(raf);
      }
    };

    resize();
    window.addEventListener('resize', resize);
    document.addEventListener('visibilitychange', onVisibility);
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
      document.removeEventListener('visibilitychange', onVisibility);
      ctx.clearRect(0, 0, 220, 220);
    };
  }, [amplitude, canvasActive, color, state]);

  const ringClass = state === 'listening' ? 'avatar-breathe' : state === 'confirming' ? 'avatar-attn' : state === 'executing' ? 'avatar-exec-glow' : state === 'follow_up' ? 'avatar-shimmer' : '';

  return (
    <div className={`avatar-root state-${state} relative grid h-[220px] w-[220px] place-items-center`} style={{ '--state-color': color } as React.CSSProperties}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" width="220" height="220" aria-hidden="true" />
      <div
        className={`relative h-[126px] w-[126px] rounded-full border-2 ${ringClass}`}
        style={{ borderColor: color, boxShadow: `0 0 24px ${color}77, inset 0 0 24px ${color}33` }}
      >
        <div className={`absolute left-1/2 top-1/2 h-[76px] w-[76px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-dashed opacity-50 ${state === 'processing' ? 'avatar-spin-reverse' : ''}`} style={{ borderColor: color }} />
        {state === 'processing' && (
          <div className="avatar-spin absolute inset-[-4px] rounded-full border-2 border-transparent" style={{ borderTopColor: color, borderRightColor: color }} />
        )}
        <div
          className={`absolute left-1/2 top-1/2 h-[18px] w-[18px] -translate-x-1/2 -translate-y-1/2 rounded-full ${state === 'processing' ? 'avatar-shimmer' : ''}`}
          style={{ background: `radial-gradient(circle, #ffffff 0%, ${color} 62%, rgba(0,0,0,0) 100%)`, boxShadow: `0 0 20px ${color}` }}
        />
      </div>
    </div>
  );
}
