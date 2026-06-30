import { getCurrentWindow } from '@tauri-apps/api/window';
import { Dashboard } from './components/dashboard/Dashboard';
import { Overlay } from './components/overlay/Overlay';
import { useJarvisSocket } from './hooks/useJarvisSocket';
import { useJarvisStore } from './stores/jarvisStore';
import { isTauri } from './lib/app';

// Under Tauri each OS window owns one view (label "overlay" | "dashboard").
// In the browser there is one window, so fall back to the in-app appView toggle.
function tauriWindowView(): 'overlay' | 'dashboard' | null {
  if (!isTauri()) return null;
  try {
    return getCurrentWindow().label === 'dashboard' ? 'dashboard' : 'overlay';
  } catch {
    return null;
  }
}

export default function App() {
  const { send } = useJarvisSocket();
  const appView = useJarvisStore((state) => state.appView);
  const view = tauriWindowView() ?? appView;

  return view === 'overlay' ? <Overlay send={send} /> : <Dashboard send={send} />;
}
