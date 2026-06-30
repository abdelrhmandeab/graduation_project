import { useJarvisStore } from '../stores/jarvisStore';

type TauriWindow = Window & {
  __TAURI_INTERNALS__?: unknown;
  __TAURI__?: unknown;
};

export function isTauri(): boolean {
  if (typeof window === 'undefined') return false;
  const w = window as TauriWindow;
  return '__TAURI_INTERNALS__' in w || '__TAURI__' in w;
}

async function invokeCommand(command: string): Promise<void> {
  const { invoke } = await import('@tauri-apps/api/core');
  await invoke(command);
}

/** Open the dashboard: a separate window under Tauri, an in-app view in the browser. */
export async function openDashboard(): Promise<void> {
  if (isTauri()) {
    await invokeCommand('open_dashboard');
    return;
  }
  useJarvisStore.getState().setAppView('dashboard');
}

/** Return from the dashboard: hide the window under Tauri, switch view in the browser. */
export async function backToOverlay(): Promise<void> {
  if (isTauri()) {
    await invokeCommand('hide_dashboard');
    return;
  }
  useJarvisStore.getState().setAppView('overlay');
}

/** Show the avatar overlay window (re-positioned bottom-right). No-op outside Tauri. */
export async function showOverlay(): Promise<void> {
  if (isTauri()) await invokeCommand('show_overlay');
}

/** Hide the avatar overlay window back to the tray. No-op outside Tauri. */
export async function hideOverlay(): Promise<void> {
  if (isTauri()) await invokeCommand('hide_overlay');
}

/** Quit the whole app (engine stays separate). No-op outside Tauri. */
export async function closeApp(): Promise<void> {
  if (!isTauri()) {
    console.warn('closeApp is only available in the Tauri runtime.');
    return;
  }
  await invokeCommand('quit_app');
}
