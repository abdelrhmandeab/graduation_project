import { useMemo, type CSSProperties, type ReactNode } from 'react';
import '../styles/gradient-background.css';

export interface GradientBackgroundProps {
  /** Content to display inside the background container. */
  children?: ReactNode;
  /** Additional CSS classes to apply to the content wrapper. */
  className?: string;
  /** Additional CSS classes to apply to the container. */
  containerClassName?: string;
  /** Array of gradient colors for the animated layers. */
  gradientColors?: string[];
  /** Opacity/intensity of the noise texture overlay (0-1). */
  noiseIntensity?: number;
  /** Speed multiplier for the gradient animation movement. */
  speed?: number;
  /** Whether to apply backdrop blur effect to the container. */
  backdropBlur?: boolean;
  /** Whether the gradient animation should be active. */
  animating?: boolean;
}

const DEFAULT_COLORS = ['rgb(255, 100, 150)', 'rgb(100, 150, 255)', 'rgb(255, 200, 100)'];

// Static SVG noise (feTurbulence) rendered once into a data URI — no per-frame
// filtering, so it stays cheap on no-GPU machines.
const NOISE_DATA_URI =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

// Base drift duration (seconds) at speed = 1. Lower speed → slower, calmer drift.
const BASE_DRIFT_SECONDS = 26;

function classNames(...parts: Array<string | undefined>): string {
  return parts.filter(Boolean).join(' ');
}

export function GradientBackground({
  children,
  className,
  containerClassName,
  gradientColors = DEFAULT_COLORS,
  noiseIntensity = 0.2,
  speed = 0.1,
  backdropBlur = false,
  animating = true,
}: GradientBackgroundProps) {
  const colors = gradientColors.length > 0 ? gradientColors : DEFAULT_COLORS;

  // One soft radial blob per color, each phase-offset so the mesh feels organic.
  const blobs = useMemo(() => {
    const safeSpeed = speed > 0 ? speed : 0.0001;
    const duration = BASE_DRIFT_SECONDS / safeSpeed;
    // Anchor positions spread around the container (top-left, bottom-right, ...).
    const anchors = [
      { left: '-25%', top: '-25%' },
      { left: '20%', top: '10%' },
      { left: '-10%', top: '25%' },
      { left: '35%', top: '-15%' },
      { left: '5%', top: '40%' },
    ];
    return colors.map((color, index) => {
      const anchor = anchors[index % anchors.length];
      const style: CSSProperties = {
        ...anchor,
        background: `radial-gradient(circle at center, ${color} 0%, transparent 62%)`,
        animationDuration: `${duration}s`,
        // Negative delay desynchronizes the layers without a startup pause.
        animationDelay: `-${(duration / colors.length) * index}s`,
      };
      return { key: `${index}-${color}`, style };
    });
  }, [colors, speed]);

  const containerStyle = backdropBlur
    ? ({ ['--jgb-backdrop-blur' as string]: '12px' } as CSSProperties)
    : undefined;

  return (
    <div className={classNames('jgb-root', containerClassName)} style={containerStyle}>
      <div className="jgb-layers" aria-hidden="true">
        {blobs.map((blob) => (
          <div
            key={blob.key}
            className={classNames('jgb-blob', animating ? 'jgb-animating' : undefined)}
            style={blob.style}
          />
        ))}
      </div>
      <div
        className="jgb-noise"
        aria-hidden="true"
        style={{
          backgroundImage: NOISE_DATA_URI,
          opacity: Math.max(0, Math.min(1, noiseIntensity)),
        }}
      />
      {backdropBlur ? <div className="jgb-backdrop" aria-hidden="true" /> : null}
      <div className={classNames('jgb-content', className)}>{children}</div>
    </div>
  );
}
