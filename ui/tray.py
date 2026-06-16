import os
import subprocess
import sys
import threading
from pathlib import Path
import webbrowser

try:
    from PIL import Image, ImageDraw
    import pystray
    from pystray import MenuItem as Item, Menu
except Exception:  # pragma: no cover
    pystray = None

from core.dialogue_manager import dialogue_manager, DialogueState
from core.logger import logger
from core.config import PROJECT_ROOT, LOG_FILE
from core.demo_mode import is_enabled as is_demo_mode_enabled
from core.demo_mode import set_enabled as set_demo_mode_enabled
from core.shutdown import perform_shutdown_cleanup


_STATE_COLORS = {
    DialogueState.IDLE: (128, 128, 128, 255),
    DialogueState.LISTENING: (0, 180, 0, 255),
    DialogueState.PROCESSING: (255, 200, 0, 255),
    DialogueState.RESPONDING: (0, 120, 255, 255),
    DialogueState.CONFIRMING: (255, 140, 0, 255),
    DialogueState.EXECUTING: (90, 90, 200, 255),
    DialogueState.FOLLOW_UP: (0, 160, 120, 255),
}


def _create_circle_icon(color, size=64):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    radius = int(size * 0.38)
    cx = cy = size // 2
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    # small border
    draw.ellipse((cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1), outline=(0,0,0,60))
    return img


class TrayManager:
    def __init__(self):
        self._icon = None
        self._thread = None
        self._running = False
        self._listener = None

    def _open_path(self, path: str) -> None:
        target = str(path)
        try:
            if os.name == 'nt':
                os.startfile(target)
            else:
                webbrowser.open(Path(target).as_uri())
        except Exception as exc:
            logger.warning("Tray: failed to open path %s: %s", target, exc)

    def _menu_open_env(self, icon, item):
        self._open_path(PROJECT_ROOT / ".env")

    def _menu_show_logs(self, icon, item):
        self._open_path(LOG_FILE)

    def _menu_open_logs_folder(self, icon, item):
        self._open_path(Path(LOG_FILE).parent)

    def _menu_restart(self, icon, item):
        try:
            command = [sys.executable, str(PROJECT_ROOT / "main.py"), *sys.argv[1:]]
            subprocess.Popen(command, cwd=str(PROJECT_ROOT), close_fds=True)
        except Exception as exc:
            logger.warning("Tray: failed to restart Jarvis: %s", exc)
            return
        try:
            perform_shutdown_cleanup()
        except Exception:
            pass
        try:
            icon.visible = False
            icon.stop()
        except Exception:
            pass
        os._exit(0)

    def _write_demo_mode_env(self, enabled: bool) -> None:
        env_path = PROJECT_ROOT / ".env"
        try:
            text = env_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("Tray: failed to read .env: %s", exc)
            return

        target_line = f"JARVIS_DEMO_MODE={'true' if enabled else 'false'}"
        lines = text.splitlines()
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith("JARVIS_DEMO_MODE="):
                lines[index] = target_line
                replaced = True
                break
        if not replaced:
            lines.append(target_line)

        newline = "\r\n" if "\r\n" in text else "\n"
        updated = newline.join(lines)
        if text.endswith(("\n", "\r")):
            updated += newline
        try:
            env_path.write_text(updated, encoding="utf-8", newline="")
        except Exception as exc:
            logger.warning("Tray: failed to update .env demo mode: %s", exc)

    def _menu_toggle_demo(self, icon, item):
        enabled = not is_demo_mode_enabled()
        set_demo_mode_enabled(enabled)
        os.environ["JARVIS_DEMO_MODE"] = "true" if enabled else "false"
        self._write_demo_mode_env(enabled)
        self._refresh_icon()

    def _menu_settings(self, icon, item):
        self._open_path(PROJECT_ROOT)

    def _menu_quit(self, icon, item):
        try:
            perform_shutdown_cleanup()
        except Exception:
            pass
        try:
            icon.visible = False
            icon.stop()
        except Exception:
            pass

    def _build_menu(self):
        return Menu(
            Item(
                'Settings',
                Menu(
                    Item('Restart Jarvis', self._menu_restart),
                    Item('Open .env', self._menu_open_env),
                    Item('Show logs', self._menu_show_logs),
                    Item('Open logs folder', self._menu_open_logs_folder),
                    Item('Toggle demo mode', self._menu_toggle_demo, checked=lambda item: is_demo_mode_enabled()),
                    Item('Open project folder', self._menu_settings),
                ),
            ),
            Item('Quit', self._menu_quit)
        )

    def _refresh_icon(self):
        icon = self._icon
        if icon is None:
            return
        try:
            state = dialogue_manager.state
            color = _STATE_COLORS.get(state, (128, 128, 128, 255))
            icon.icon = _create_circle_icon(color)
            icon.title = f"Jarvis — {state.value}"
            try:
                icon.update_menu()
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Tray refresh failed: %s", exc)

    def _on_state_changed(self, old_state, new_state):
        _ = old_state
        if not self._running:
            return
        self._refresh_icon()

    def start(self):
        if pystray is None:
            logger.info("pystray not available; tray icon disabled.")
            return
        if self._running:
            return
        self._running = True
        # initial icon
        img = _create_circle_icon(_STATE_COLORS.get(DialogueState.IDLE))
        menu = self._build_menu()
        icon = pystray.Icon("jarvis_tray", img, "Jarvis", menu)
        self._icon = icon
        self._listener = self._on_state_changed
        dialogue_manager.register_state_listener(self._listener)
        self._refresh_icon()

        def target():
            try:
                icon.run()
            except Exception as exc:
                logger.warning("Tray icon failed: %s", exc)
            finally:
                try:
                    if self._listener is not None:
                        dialogue_manager.unregister_state_listener(self._listener)
                except Exception:
                    pass

        self._thread = threading.Thread(target=target, name="jarvis-tray", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            if self._icon:
                self._icon.visible = False
                self._icon.stop()
        except Exception:
            pass
        try:
            if self._listener is not None:
                dialogue_manager.unregister_state_listener(self._listener)
        except Exception:
            pass


_tray_manager = TrayManager()


def start_tray():
    _tray_manager.start()


def stop_tray():
    _tray_manager.stop()
