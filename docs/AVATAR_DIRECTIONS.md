# Jarvis Avatar — 4 Design Directions (Premium Glass)

Four selectable avatar identities. All share the 7-state color language (`STATE_COLORS`
in `desktop/src/protocol.ts`) and the no-GPU animation rules (transform + opacity only,
Canvas 2D for audio visualization, no WebGL). Each direction decides where the state color
appears and how it expresses motion. The look must read as **premium, translucent glass** —
not flat neon.

State colors (shared):

| State | Hex |
|-------|-----|
| idle | `#808080` |
| listening | `#00B400` |
| processing | `#FFC800` |
| responding | `#0078FF` |
| confirming | `#FF8C00` |
| executing | `#5A5AC8` |
| follow_up | `#00A078` |

---

## Direction A — Aurora (glass sphere)

**Form & silhouette.** A perfect sphere. No edges, seams, or mechanical reference — a pure
volumetric form, identical from every angle: a circle. The most abstract direction; it
radiates rather than represents.

**Material & surface.** Neither solid nor liquid. A three-stop radial gradient lit from the
upper-left quadrant gives genuine volumetric depth: a near-white specular bloom at 38%
horizontal / 32% vertical, transitioning through the full-saturation state color at
mid-sphere, then compressing into a darkened shadow tone at the base. A second inner layer —
a tight white radial highlight floating slightly off-center — reads as a lens reflection /
LED hotspot through diffusion. Effect: wet, pressurized, alive; emitted from within.

**Light & glow.** Two concentric halos bleed outward from the surface edge: a tighter bloom
at the configured glow radius, and a wider, dramatically dimmer secondary at double that
radius — the optical consequence of light behind thin glass. Softly tints the surrounding
desktop.

**Canvas-2D waveform.** LISTENING: three concentric wobbling rings expand/contract with mic
amplitude, drawn on a Canvas layer **behind** the orb so they radiate through the glass. Each
ring carries a low-frequency sine wobble on its radial distance (ripples through thick
liquid). RESPONDING: a horizontal sinusoidal wave passes through the orb's equatorial plane,
tapered at both edges via a sin-envelope. PROCESSING: a conic-gradient sweep masked to a thin
ring — a single luminous arc rotating steadily, like a radar trace.

**State color philosophy.** State color **is** the orb — gradient, glow, shadow, halo all
recalculate from `--orb-color`. 0.5s ease on color change reads as mood, not mode switch.

**Motion language.** CSS transforms only: `breathe` (scale oscillation, LISTENING/RESPONDING),
`attn` (larger scale pulse, CONFIRMING), `execGlow` (brightness oscillation via CSS filter,
EXECUTING), `shimmer` (opacity oscillation, FOLLOW_UP), spin sweep ring (PROCESSING). No
position/layout changes.

**Personality.** The presence direction — something natural that happens to be intelligent.
"Something woke up," not "app opened." Refs: the orb in *Arrival*, a distant lighthouse.

---

## Direction B — Glyph (technical ring)

**Form & silhouette.** A hollow luminous ring — mass moved entirely to the perimeter. Center
is open dark space with a small dot-core. Two concentric circles + one point of light. A
symbol, not a shape. Refs: Nothing Phone glyph LEDs, Braun restraint, instrument reticles.

**Material & surface.** No physical material. Outer ring is a 2px luminous stroke — sharp,
anti-aliased, color-accurate — with soft ambient glow spilling inward/outward (a large blurred
box-shadow on the ring element, not a drawn ring). A secondary inner ring at ~60% radius is a
dashed 1.5px stroke at half opacity (engineering-drawing depth reference). The dot-core is a
tight white-to-state-color radial gradient with glow blur: a pinhole light source. Drawn in
light, not built from material.

**Canvas-2D dot matrix.** LISTENING: 28 discrete dots in a precise circle at 1.5× ring radius
scale individually with the amplitude envelope; each dot's sine phase is offset by its index,
so the ring breathes as a traveling wave (polar oscilloscope). RESPONDING: the 28 dots light
sequentially as a traveling wave (circular equalizer). PROCESSING: a single arc rotates on the
ring (border-top + border-right CSS trick); the inner dashed ring counter-rotates at a
different speed.

**State color philosophy.** Almost no fill area — color lives in stroke luminance and glow
spread. Signal-like, not atmospheric (LED status indicator vs mood lamp). High, immediate
contrast between states; legible in peripheral vision.

**Motion language.** Transform-only: outer ring breathes (scale, LISTENING), dot-core shimmer
(PROCESSING), spin arc rotates (PROCESSING), inner ring counter-rotates. Animations scoped to
`state-*` class selectors so they switch cleanly.

**Personality.** The technical direction — developer tools, system monitors, spectrum
analyzers. No emotional warmth; a signal, not a face. Refs: Nothing glyph matrix, oscilloscope,
LCARS.

---

## Direction C — Glass AI (intelligence core)

**Form & silhouette.** A rounded square — the app-icon archetype made physical. Generous
superellipse corner radius (between circle and square): friendly yet stable. "I am an object
with a purpose. I live on your desktop."

**Material & surface.** True glassmorphism, restrained. Body is a three-stop linear gradient
from ice-white (`#EAFBFF` @ 24% opacity) → mist-blue (`#D9ECF5` @ 13%) → soft steel
(`#B7C8D8` @ 20%): a translucent panel angled to studio light. A 1.6px cyan-white rim stroke
@ 55% opacity defines the edge. An inner border at 82% scale (1px, 32% opacity) adds glass
thickness. A diagonal highlight streak across the top-left quadrant is the strongest specular
cue.

**Lightbulb intelligence core.** Centered minimalist single-weight line lightbulb: circular
bulb body, curved base-cap arc, two horizontal screw-thread lines, pointed tip, W-shaped
filament. Strokes in state color, 1.7px round-capped; filament 1.4px fully opaque. Surrounded
by a soft radial glow that pulses with state, radiating through the frosted glass. Lightbulb =
idea / intelligence as an illumination event.

**Neural circuit pathways.** 12 paths radiate from the core to 16 terminal nodes at corners,
edges, midpoints. Base traces nearly invisible (stroke-opacity 0.18). Animated pulse layer uses
`stroke-dasharray: 14 100` traveling along each path with per-trace stagger, so light travels
from core outward, arrives at a node, repeats. Per state: LISTENING flows inward; RESPONDING
flows outward; PROCESSING races at 1× speed; IDLE drifts slowly. The 16 nodes twinkle
independently.

**State color philosophy.** The glass body never changes. Only internal illumination shifts —
ambient halo, circuit trace color, bulb stroke, core glow track `--bot-active`. At IDLE,
`--bot-active` is hardcoded to `#AEEFFF` (ice blue). Permanent surface vs transient
illumination is the key idea.

**Motion language.** Tile floats with a gentle 4.6s ease-in-out bob (6px vertical, 0.6°
rotation). Core breathes at state-tuned rates. A 4-point sparkle (top-right) pulses on its own
5s cycle. Cool-toned drop shadow (`#6D7886`).

**Personality.** The product-identity direction — could be an app icon / splash / marketing
visual. Premium material weight + lightbulb legibility + circuit depth. Between B's precision
and D's warmth. Refs: Iron Man arc reactor, Nothing ecosystem, Teenage Engineering.

---

## Direction D — Friendly AI (companion face)

**Form & silhouette.** Same rounded-square app-icon archetype as C, but the interior is a
face. Two-layer architecture: outer shell + inner face panel = a screen within a frame. Inner
panel radius (rx=18) slightly tighter than outer (rx=30) for visible material depth. A square
with a face — a desktop companion mascot.

**Material & surface.** Outer frame uses C's frosted glass gradient (ice-white → mist-blue →
soft steel) with 1.6px cyan-white rim + 1px inner edge glow. Inner face panel uses a slightly
different, more transparent gradient shifted toward `#B7C8D8` at the base, so it reads as a
separate inset surface — two distinct panes of glass.

**The face.** Exactly three strokes: two eye arcs + one mouth arc. Each is a single quadratic
Bézier, 2.6px (eyes) / 2.1px (mouth), round-capped, state color. No irises, pupils, eyelids,
nose, or eyebrows — nothing pushing toward realism or specificity. Default closed-smiling
(downward-bowing ˘) arcs = universal warmth.

**Per-state expressions** (snap eye + mouth `d` attributes):

| State | Eyes | Mouth |
|-------|------|-------|
| idle | downward bowing arcs — smiling closed | full gentle smile |
| listening | upward bowing arcs — wide open, attentive | smaller attentive smile |
| processing | flat horizontal lines — scanning | neutral flat line |
| responding | gentle downward arcs (speaking squint) | full smile, mouth animates open/close |
| confirming | slightly raised arcs — alert, questioning | small arc, slightly tightened |
| executing | focused arcs — steady | calm smile |
| follow_up | same as idle — curious | happy full smile |

**Eye glow & bloom.** Behind each eye, a semi-transparent ellipse (rx=11, ry=8) filled with
the state color @ 22% opacity, through a `feGaussianBlur stdDeviation=3` bloom — a soft colored
halo that pulses with the CSS animations. The eyes glow; the face is internally lit.

**Periodic blink.** At IDLE, eye groups animate `scaleY(0.06)` at 91% of a 5.2s cycle —
collapsing each eye to a thin line, recovering over ~200ms. The single most humanizing decision.

**Supporting elements.** A 4-point sparkle (top-right of inner panel) pulses on a 5.2s cycle.
Three status dots at the inner panel's bottom edge animate with staggered per-state timing
(slow twinkle idle, traveling wave listening, rapid beat responding). Six subtle circuit traces
(stroke-opacity 0.11 base, 0.28–0.82 pulse per state) ground the character technologically
without competing with the face.

**State color philosophy.** Same as C — glass body never changes, internal illumination shifts —
but color also lives in the face strokes: green eyes listening, amber thinking, blue speaking,
orange confirming. Mood made literal.

**Personality.** The companion direction — greeted, not addressed. Sage + Caregiver. Emotionally
present without being anthropomorphic. Refs: Arc System Works character design, Tamagotchi face
language, Nothing-meets-Sony.
