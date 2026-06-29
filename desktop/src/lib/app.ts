type TauriWindow = Window & {
  __TAURI_INTERNALS__?: unknown;
  __TAURI__?: unknown;
};

export async function closeApp() {
  const currentWindow = window as TauriWindow;
  const isTauri = '__TAURI_INTERNALS__' in currentWindow || '__TAURI__' in currentWindow;

  if (!isTauri) {
    console.warn('closeApp is only available in the Tauri runtime.');
    return;
  }

  const { getCurrentWindow } = await import('@tauri-apps/api/window');
  await getCurrentWindow().close();
}
