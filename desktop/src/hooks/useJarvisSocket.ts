import { useCallback, useEffect, useRef } from 'react';
import { JARVIS_WS_URL, type EngineEvent, type UICommand } from '../protocol';
import { useJarvisStore } from '../stores/jarvisStore';

export function useJarvisSocket(): { send: (cmd: UICommand) => void; connectionStatus: string } {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectDelayRef = useRef(1000);
  const shouldReconnectRef = useRef(true);
  const dispatch = useJarvisStore((state) => state.dispatch);
  const connectionStatus = useJarvisStore((state) => state.connectionStatus);
  const setConnectionStatus = useJarvisStore((state) => state.setConnectionStatus);

  useEffect(() => {
    const clearReconnectTimer = () => {
      if (reconnectTimerRef.current !== null) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const connect = () => {
      clearReconnectTimer();
      setConnectionStatus('connecting');

      const socket = new WebSocket(JARVIS_WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        reconnectDelayRef.current = 1000;
        setConnectionStatus('connected');
      };

      socket.onmessage = (message) => {
        try {
          dispatch(JSON.parse(message.data as string) as EngineEvent);
        } catch (error) {
          if (!(error instanceof SyntaxError)) {
            throw error;
          }

          dispatch({ type: 'error', message: 'Received invalid JSON from Jarvis engine.' });
        }
      };

      socket.onerror = () => {
        setConnectionStatus('disconnected');
      };

      socket.onclose = () => {
        setConnectionStatus('disconnected');
        socketRef.current = null;

        if (!shouldReconnectRef.current) {
          return;
        }

        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, 8000);
        reconnectTimerRef.current = window.setTimeout(connect, delay);
      };
    };

    shouldReconnectRef.current = true;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      clearReconnectTimer();
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [dispatch, setConnectionStatus]);

  const send = useCallback((cmd: UICommand) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  return { send, connectionStatus };
}
