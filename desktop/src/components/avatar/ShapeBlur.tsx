import { useEffect, useRef, useState, type CSSProperties } from 'react';
import * as THREE from 'three';

const vertexShader = /* glsl */ `
varying vec2 v_texcoord;
void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    v_texcoord = uv;
}
`;

const fragmentShader = /* glsl */ `
varying vec2 v_texcoord;

uniform vec2 u_mouse;
uniform vec2 u_resolution;
uniform float u_pixelRatio;
uniform vec3 u_color;

uniform float u_shapeSize;
uniform float u_roundness;
uniform float u_borderSize;
uniform float u_circleSize;
uniform float u_circleEdge;

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif
#ifndef TWO_PI
#define TWO_PI 6.2831853071795864769252867665590
#endif

#ifndef VAR
#define VAR 0
#endif

#ifndef FNC_COORD
#define FNC_COORD
vec2 coord(in vec2 p) {
    p = p / u_resolution.xy;
    if (u_resolution.x > u_resolution.y) {
        p.x *= u_resolution.x / u_resolution.y;
        p.x += (u_resolution.y - u_resolution.x) / u_resolution.y / 2.0;
    } else {
        p.y *= u_resolution.y / u_resolution.x;
        p.y += (u_resolution.x - u_resolution.y) / u_resolution.x / 2.0;
    }
    p -= 0.5;
    p *= vec2(-1.0, 1.0);
    return p;
}
#endif

#define st0 coord(gl_FragCoord.xy)
#define mx coord(u_mouse * u_pixelRatio)

float sdRoundRect(vec2 p, vec2 b, float r) {
    vec2 d = abs(p - 0.5) * 4.2 - b + vec2(r);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)) - r;
}
float sdCircle(in vec2 st, in vec2 center) {
    return length(st - center) * 2.0;
}
float sdPoly(in vec2 p, in float w, in int sides) {
    float a = atan(p.x, p.y) + PI;
    float r = TWO_PI / float(sides);
    float d = cos(floor(0.5 + a / r) * r - a) * length(max(abs(p) * 1.0, 0.0));
    return d * 2.0 - w;
}

float aastep(float threshold, float value) {
    float afwidth = length(vec2(dFdx(value), dFdy(value))) * 0.70710678118654757;
    return smoothstep(threshold - afwidth, threshold + afwidth, value);
}
float fill(in float x) { return 1.0 - aastep(0.0, x); }
float fill(float x, float size, float edge) {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}
float stroke(in float d, in float t) { return (1.0 - aastep(t, abs(d))); }
float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5);
    return clamp(d, 0.0, 1.0);
}

float strokeAA(float x, float size, float w, float edge) {
    float afwidth = length(vec2(dFdx(x), dFdy(x))) * 0.70710678;
    float d = smoothstep(size - edge - afwidth, size + edge + afwidth, x + w * 0.5)
            - smoothstep(size - edge - afwidth, size + edge + afwidth, x - w * 0.5);
    return clamp(d, 0.0, 1.0);
}

void main() {
    vec2 st = st0 + 0.5;
    vec2 posMouse = mx * vec2(1., -1.) + 0.5;

    float size = u_shapeSize;
    float roundness = u_roundness;
    float borderSize = u_borderSize;
    float circleSize = u_circleSize;
    float circleEdge = u_circleEdge;

    float sdfCircle = fill(
        sdCircle(st, posMouse),
        circleSize,
        circleEdge
    );

    float sdf;
    if (VAR == 0) {
        sdf = sdRoundRect(st, vec2(size), roundness);
        sdf = strokeAA(sdf, 0.0, borderSize, sdfCircle) * 4.0;
    } else if (VAR == 1) {
        sdf = sdCircle(st, vec2(0.5));
        sdf = fill(sdf, 0.6, sdfCircle) * 1.2;
    } else if (VAR == 2) {
        sdf = sdCircle(st, vec2(0.5));
        sdf = strokeAA(sdf, 0.58, 0.02, sdfCircle) * 4.0;
    } else if (VAR == 3) {
        sdf = sdPoly(st - vec2(0.5, 0.45), 0.3, 3);
        sdf = fill(sdf, 0.05, sdfCircle) * 1.4;
    }

    vec3 color = u_color;
    float alpha = sdf;
    gl_FragColor = vec4(color.rgb, alpha);
}
`;

interface ShapeBlurProps {
  className?: string;
  variation?: number;
  pixelRatioProp?: number;
  shapeSize?: number;
  roundness?: number;
  borderSize?: number;
  circleSize?: number;
  circleEdge?: number;
  /** Shape color (hex). Defaults to white to match the original component. */
  color?: string;
  /** Speed of the automatic reveal orbit when the mouse is idle (0 = none). */
  autoSpeed?: number;
  /** When this value changes, a one-shot reveal pulse plays (e.g. on state change). */
  pulseKey?: string | number;
  /** When false (reduced motion) the render loop is paused after one frame. */
  animate?: boolean;
  /** Rendered if no WebGL context is available. */
  fallback?: React.ReactNode;
  style?: CSSProperties;
}

const ShapeBlur = ({
  className = '',
  variation = 0,
  pixelRatioProp = 2,
  shapeSize = 1.2,
  roundness = 0.4,
  borderSize = 0.05,
  circleSize = 0.3,
  circleEdge = 0.5,
  color = '#ffffff',
  autoSpeed = 0.9,
  pulseKey,
  animate = true,
  fallback = null,
  style,
}: ShapeBlurProps) => {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const colorRef = useRef(new THREE.Color(color));
  const animateRef = useRef(animate);
  const autoSpeedRef = useRef(autoSpeed);
  const mouseActiveUntilRef = useRef(0);
  const pulseStartRef = useRef(-10);
  const [failed, setFailed] = useState(false);

  // Keep mutable values current without re-running the heavy WebGL effect.
  colorRef.current.set(color);
  animateRef.current = animate;
  autoSpeedRef.current = autoSpeed;

  // Trigger a reveal pulse whenever pulseKey changes (e.g. dialogue state change).
  useEffect(() => {
    pulseStartRef.current = performance.now() * 0.001;
  }, [pulseKey]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let active = true;
    let animationFrameId = 0;
    let time = 0;
    let lastTime = 0;
    let running = !document.hidden;

    const vMouse = new THREE.Vector2();
    const vMouseDamp = new THREE.Vector2();
    const vResolution = new THREE.Vector2();

    let w = 1;
    let h = 1;

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera();
    camera.position.z = 1;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    } catch {
      setFailed(true);
      return;
    }
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const geo = new THREE.PlaneGeometry(1, 1);
    const material = new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        u_mouse: { value: vMouseDamp },
        u_resolution: { value: vResolution },
        u_pixelRatio: { value: pixelRatioProp },
        u_color: { value: new THREE.Color(colorRef.current) },
        u_shapeSize: { value: shapeSize },
        u_roundness: { value: roundness },
        u_borderSize: { value: borderSize },
        u_circleSize: { value: circleSize },
        u_circleEdge: { value: circleEdge },
      },
      defines: { VAR: variation },
      transparent: true,
    });

    const quad = new THREE.Mesh(geo, material);
    scene.add(quad);

    const onPointerMove = (e: PointerEvent | MouseEvent) => {
      const rect = mount.getBoundingClientRect();
      vMouse.set(e.clientX - rect.left, e.clientY - rect.top);
      // Hover takes over for a short window; afterwards the auto-orbit resumes.
      mouseActiveUntilRef.current = performance.now() * 0.001 + 1.5;
    };

    document.addEventListener('mousemove', onPointerMove);
    document.addEventListener('pointermove', onPointerMove);

    const resize = () => {
      if (!active) return;
      w = mount.clientWidth;
      h = mount.clientHeight;
      const dpr = Math.min(window.devicePixelRatio, 2);

      renderer.setSize(w, h);
      renderer.setPixelRatio(dpr);

      camera.left = -w / 2;
      camera.right = w / 2;
      camera.top = h / 2;
      camera.bottom = -h / 2;
      camera.updateProjectionMatrix();

      quad.scale.set(w, h, 1);
      vResolution.set(w, h).multiplyScalar(dpr);
      material.uniforms.u_pixelRatio.value = dpr;
    };

    resize();
    window.addEventListener('resize', resize);

    const ro = new ResizeObserver(() => {
      if (!active) return;
      resize();
    });
    ro.observe(mount);

    const baseCircleSize = circleSize;

    const renderFrame = (t: number, dt: number) => {
      // Target the real mouse while hovering; otherwise auto-orbit the reveal
      // point so the shape animates on its own (and on state change).
      const mouseActive = t < mouseActiveUntilRef.current;
      let tx: number;
      let ty: number;
      if (mouseActive) {
        tx = vMouse.x;
        ty = vMouse.y;
      } else {
        const orbit = Math.min(w, h) * 0.32;
        const sp = autoSpeedRef.current;
        tx = w / 2 + Math.cos(t * sp) * orbit;
        ty = h / 2 + Math.sin(t * sp * 1.3) * orbit;
      }
      vMouseDamp.x = THREE.MathUtils.damp(vMouseDamp.x, tx, 8, dt);
      vMouseDamp.y = THREE.MathUtils.damp(vMouseDamp.y, ty, 8, dt);

      // One-shot reveal bloom that decays after each state change.
      const elapsed = t - pulseStartRef.current;
      const pulse = Math.max(0, Math.exp(-elapsed * 3.5)) * 0.45;
      material.uniforms.u_circleSize.value = baseCircleSize + pulse;

      material.uniforms.u_color.value.set(colorRef.current);
      renderer.render(scene, camera);
    };

    const update = () => {
      if (!active) return;
      if (!running || !animateRef.current) return;
      time = performance.now() * 0.001;
      const dt = time - lastTime;
      lastTime = time;
      renderFrame(time, dt);
      animationFrameId = requestAnimationFrame(update);
    };

    const onVisibility = () => {
      running = !document.hidden;
      cancelAnimationFrame(animationFrameId);
      if (running && animateRef.current) {
        lastTime = performance.now() * 0.001;
        animationFrameId = requestAnimationFrame(update);
      }
    };
    document.addEventListener('visibilitychange', onVisibility);

    if (animate) {
      update();
    } else {
      renderFrame(performance.now() * 0.001, 0);
    }

    return () => {
      active = false;
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
      ro.disconnect();
      document.removeEventListener('mousemove', onPointerMove);
      document.removeEventListener('pointermove', onPointerMove);
      document.removeEventListener('visibilitychange', onVisibility);
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement);
      }
      geo.dispose();
      material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [variation, pixelRatioProp, shapeSize, roundness, borderSize, circleSize, circleEdge, animate]);

  if (failed) {
    return (
      <div className={className} style={{ width: '100%', height: '100%', ...style }}>
        {fallback}
      </div>
    );
  }

  return <div className={className} ref={mountRef} style={{ width: '100%', height: '100%', ...style }} />;
};

export default ShapeBlur;
