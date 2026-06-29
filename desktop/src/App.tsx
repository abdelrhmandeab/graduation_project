import { Dashboard } from './components/dashboard/Dashboard';
import { Overlay } from './components/overlay/Overlay';
import { useJarvisSocket } from './hooks/useJarvisSocket';
import { useJarvisStore } from './stores/jarvisStore';

export default function App() {
  const { send } = useJarvisSocket();
  const appView = useJarvisStore((state) => state.appView);

  return appView === 'overlay' ? <Overlay send={send} /> : <Dashboard send={send} />;
}
