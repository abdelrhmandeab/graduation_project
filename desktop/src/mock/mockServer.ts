import type { Plugin } from 'vite';
import { WebSocket, WebSocketServer } from 'ws';
import { MOCK_WS_PORT, type DialogueState, type EngineEvent, type TextCommandMessage } from '../protocol';

type Cleanup = () => void;

const wait = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function send(socket: WebSocket, event: EngineEvent) {
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(event));
  }
}

function schedule(cleanups: Cleanup[], callback: () => void, ms: number) {
  const timeout = setTimeout(callback, ms);
  cleanups.push(() => clearTimeout(timeout));
}

function sendState(socket: WebSocket, state: DialogueState) {
  send(socket, { type: 'state_changed', state });
}

function startConversationLoop(socket: WebSocket) {
  let closed = false;
  const cleanups: Cleanup[] = [];

  socket.on('close', () => {
    closed = true;
    cleanups.splice(0).forEach((cleanup) => cleanup());
  });

  const run = async () => {
    while (!closed && socket.readyState === WebSocket.OPEN) {
      sendState(socket, 'idle');
      await wait(2000);
      if (closed) {
        break;
      }

      sendState(socket, 'listening');
      const amplitudeInterval = setInterval(() => {
        send(socket, { type: 'amplitude', level: 0.15 + Math.random() * 0.7 });
      }, 67);
      cleanups.push(() => clearInterval(amplitudeInterval));

      const partials = ['open', 'open the', 'open the browser', 'open the browser please'];
      partials.forEach((text, index) => {
        schedule(cleanups, () => {
          send(socket, { type: 'partial_transcript', text, language: 'en' });
        }, (index + 1) * 500);
      });

      await wait(3000);
      clearInterval(amplitudeInterval);
      if (closed) {
        break;
      }

      sendState(socket, 'processing');
      send(socket, { type: 'final_transcript', text: 'open the browser please', language: 'en' });
      await wait(1500);
      if (closed) {
        break;
      }

      sendState(socket, 'responding');
      send(socket, { type: 'response', text: 'Opening Chrome for you.', language: 'en' });
      await wait(2000);
    }
  };

  void run();
}

export function mockJarvisServer(): Plugin {
  let wsServer: WebSocketServer | null = null;

  return {
    name: 'jarvis-mock-server',
    configureServer() {
      if (wsServer) {
        return;
      }

      wsServer = new WebSocketServer({ port: MOCK_WS_PORT });
      console.log(`[mock] Jarvis mock server listening on ws://localhost:${MOCK_WS_PORT}`);

      wsServer.on('connection', (socket) => {
        startConversationLoop(socket);

        socket.on('message', (rawMessage) => {
          let message: TextCommandMessage | null = null;
          try {
            message = JSON.parse(rawMessage.toString()) as TextCommandMessage;
          } catch (error) {
            if (!(error instanceof SyntaxError)) {
              throw error;
            }

            send(socket, { type: 'error', message: 'Invalid command JSON.' });
            return;
          }

          if (message.type === 'text_command') {
            sendState(socket, 'processing');
            setTimeout(() => {
              send(socket, { type: 'response', text: `You said: ${message.text}`, language: 'en' });
              sendState(socket, 'idle');
            }, 1000);
          }
        });
      });
    },
  };
}
