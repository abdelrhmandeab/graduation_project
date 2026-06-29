import { Avatar } from './components/Avatar';
import { DirectionPicker } from './components/DirectionPicker';
import { GradientBackground } from './components/GradientBackground';
import { Transcript } from './components/Transcript';
import { useJarvisSocket } from './hooks/useJarvisSocket';

// Fixed premium backdrop palette (does not change with assistant state).
const BACKGROUND_COLORS = ['#0078FF', '#8EEBFF', '#0B1430'];

export default function App() {
  useJarvisSocket();

  return (
    <GradientBackground
      containerClassName="min-h-screen bg-[#0A0A0F] text-white"
      className="flex min-h-screen items-center justify-center"
      gradientColors={BACKGROUND_COLORS}
      noiseIntensity={0.18}
      speed={0.1}
    >
      <main className="flex flex-col items-center gap-6">
        <h1 className="text-3xl font-semibold tracking-normal">Jarvis</h1>
        <Avatar />
        <Transcript />
      </main>
      <DirectionPicker />
    </GradientBackground>
  );
}
