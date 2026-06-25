"""Optional WebSocket bridge between Jarvis engine events and desktop UI clients."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from core import config
from core.dialogue_manager import dialogue_manager
from core.logger import get_logger
from ui.events import (
    COMMAND_CONFIG_REQUEST,
    COMMAND_FEATURE_FLAG,
    COMMAND_HEALTH_REQUEST,
    COMMAND_MUTE_TOGGLE,
    COMMAND_SETTING_UPDATE,
    COMMAND_TEXT,
    EVENT_CONFIG,
    EVENT_ERROR,
    EVENT_HEALTH,
    EVENT_RESPONSE,
    EVENT_STATE_CHANGED,
    make_event,
    to_json,
)

logger = get_logger("bridge")

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent.
    FastAPI = None
    WebSocket = Any
    WebSocketDisconnect = Exception
    uvicorn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class JarvisBridge:
    def __init__(self) -> None:
        self.clients = set()
        self.lock = threading.Lock()
        self.loop = None
        self.muted = False
        self.enabled = bool(getattr(config, "UI_BRIDGE_ENABLED", True))
        self.host = str(getattr(config, "UI_BRIDGE_HOST", "127.0.0.1") or "127.0.0.1")
        self.port = int(getattr(config, "UI_BRIDGE_PORT", 9720) or 9720)
        self._running = False
        self._state_listener = None
        self._server_thread = None
        self._start_lock = threading.Lock()
        self._missing_deps_logged = False

    @property
    def available(self) -> bool:
        return FastAPI is not None and uvicorn is not None

    @property
    def running(self) -> bool:
        return bool(self._running and self.loop is not None)

    def start(self) -> None:
        if not self.enabled:
            logger.info("UI bridge disabled.")
            return
        if not self.available:
            if not self._missing_deps_logged:
                logger.info("UI bridge optional dependencies unavailable; bridge disabled: %s", _IMPORT_ERROR)
                self._missing_deps_logged = True
            return

        with self._start_lock:
            if self._running:
                return
            app = self._create_app()
            self._state_listener = self._on_state_changed
            try:
                dialogue_manager.register_state_listener(self._state_listener)
            except Exception:
                logger.debug("Failed to register bridge state listener", exc_info=True)

            self._server_thread = threading.Thread(
                target=self._run_server,
                args=(app,),
                name="jarvis-ui-bridge",
                daemon=True,
            )
            self._running = True
            self._server_thread.start()
            logger.info("UI bridge starting on ws://%s:%s/ws", self.host, self.port)

    def _create_app(self):
        app = FastAPI()

        @app.on_event("startup")
        async def _on_startup() -> None:
            self.loop = asyncio.get_running_loop()

        @app.websocket("/ws")
        async def _websocket_endpoint(websocket: WebSocket) -> None:
            await self.handle_client(websocket)

        return app

    def _run_server(self, app) -> None:
        try:
            uvicorn.run(app, host=self.host, port=self.port, log_level="info")
        except Exception:
            logger.exception("UI bridge server failed")
        finally:
            self._running = False
            self.loop = None
            try:
                if self._state_listener is not None:
                    dialogue_manager.unregister_state_listener(self._state_listener)
            except Exception:
                logger.debug("Failed to unregister bridge state listener", exc_info=True)

    async def handle_client(self, websocket: WebSocket) -> None:
        try:
            await websocket.accept()
            self._register_client(websocket)
            await websocket.send_text(to_json(self._config_event()))
            while True:
                try:
                    raw_message = await websocket.receive_text()
                    await self._handle_message(websocket, raw_message)
                except WebSocketDisconnect:
                    break
                except Exception:
                    logger.exception("UI bridge websocket message failed")
                    try:
                        await websocket.send_text(
                            to_json(
                                make_event(
                                    EVENT_ERROR,
                                    message="Bridge failed to process websocket message.",
                                    recoverable=True,
                                )
                            )
                        )
                    except Exception:
                        logger.debug("Failed to send bridge error event", exc_info=True)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("UI bridge websocket client failed")
        finally:
            self._unregister_client(websocket)

    async def _handle_message(self, websocket: WebSocket, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await websocket.send_text(
                to_json(make_event(EVENT_ERROR, message="Invalid JSON command.", recoverable=True))
            )
            return

        command_type = str(message.get("type") or "")
        if command_type == COMMAND_TEXT:
            text = str(message.get("text") or "").strip()
            language = message.get("language")
            if not text:
                return
            response = await asyncio.to_thread(self._route_text_command, text, language)
            self.broadcast(make_event(EVENT_RESPONSE, text=str(response or ""), language=language))
            return

        if command_type == COMMAND_MUTE_TOGGLE:
            self.muted = bool(message.get("muted"))
            logger.info("UI bridge mute toggled: %s", self.muted)
            return

        if command_type == COMMAND_CONFIG_REQUEST:
            await websocket.send_text(to_json(self._config_event()))
            return

        if command_type == COMMAND_HEALTH_REQUEST:
            self.broadcast(self._health_event())
            return

        if command_type == COMMAND_SETTING_UPDATE:
            # TODO: Apply runtime-safe setting updates in a later bridge phase.
            logger.info("UI bridge setting_update received: key=%s", message.get("key"))
            return

        if command_type == COMMAND_FEATURE_FLAG:
            # TODO: Apply runtime-safe feature flag updates in a later bridge phase.
            logger.info(
                "UI bridge feature_flag received: flag=%s enabled=%s",
                message.get("flag"),
                message.get("enabled"),
            )
            return

        logger.info("UI bridge ignored unknown command type: %s", command_type)

    def _route_text_command(self, text: str, language) -> str:
        try:
            from core.command_router import route_command

            return route_command(text, detected_language=language)
        except Exception:
            logger.exception("UI bridge text_command routing failed")
            return "I could not process that command."

    def _register_client(self, websocket: WebSocket) -> None:
        with self.lock:
            self.clients.add(websocket)
        logger.info("UI bridge client connected; clients=%s", len(self.clients))

    def _unregister_client(self, websocket: WebSocket) -> None:
        with self.lock:
            self.clients.discard(websocket)
        logger.info("UI bridge client disconnected; clients=%s", len(self.clients))

    def _on_state_changed(self, old_state, new_state) -> None:
        try:
            self.broadcast(
                make_event(
                    EVENT_STATE_CHANGED,
                    state=getattr(new_state, "value", str(new_state)),
                    previous=getattr(old_state, "value", str(old_state)),
                )
            )
        except Exception:
            logger.debug("Bridge state listener failed", exc_info=True)

    def _config_event(self) -> dict:
        values = {
            "model": getattr(config, "LLM_MODEL", ""),
            "model_tier": "auto" if bool(getattr(config, "LLM_AUTO_SELECT_MODEL", False)) else "configured",
            "wake_mode": getattr(config, "WAKE_WORD_MODE", ""),
            "feature_flags": dict(getattr(config, "FEATURE_FLAGS", {}) or {}),
            "stt_backend": getattr(config, "STT_BACKEND", ""),
            "tts_backend": getattr(config, "TTS_DEFAULT_BACKEND", ""),
            "persona": getattr(config, "PERSONA_DEFAULT", ""),
        }
        return make_event(EVENT_CONFIG, values=values)

    def _health_event(self) -> dict:
        # TODO: Replace this cheap bridge-local status with collect_diagnostics()
        # once the UI can tolerate slower health probes off the socket hot path.
        checks = [
            {
                "name": "ui_bridge",
                "status": "ok" if self.running else "degraded",
                "detail": f"clients={len(self.clients)} muted={self.muted}",
            }
        ]
        return make_event(EVENT_HEALTH, checks=checks)

    def broadcast(self, event: dict) -> None:
        if not self.running:
            return
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(event), loop)
        except Exception:
            logger.debug("Failed to schedule bridge broadcast", exc_info=True)

    async def _broadcast(self, event: dict) -> None:
        payload = to_json(event)
        with self.lock:
            clients = tuple(self.clients)
        stale_clients = []
        for websocket in clients:
            try:
                await websocket.send_text(payload)
            except Exception:
                logger.debug("Failed to send bridge event to client", exc_info=True)
                stale_clients.append(websocket)
        if stale_clients:
            with self.lock:
                for websocket in stale_clients:
                    self.clients.discard(websocket)


bridge = JarvisBridge()


def start_bridge() -> None:
    bridge.start()


def broadcast_event(event: dict) -> None:
    bridge.broadcast(event)
