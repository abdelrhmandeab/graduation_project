import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { DialogueState, EngineEvent, Language } from '../protocol';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';
export type AvatarDirection = 'aurora' | 'glyph' | 'glassai' | 'companion';

interface JarvisState {
  connectionStatus: ConnectionStatus;
  dialogueState: DialogueState;
  amplitude: number;
  muted: boolean;
  partialTranscript: string;
  finalTranscript: string;
  transcriptLanguage: Language | null;
  response: string;
  responseLanguage: Language | null;
  stages: Array<{ name: string; duration_ms: number }>;
  doctor: { ok: boolean; checks: Array<{ name: string; ok: boolean; details: string }> } | null;
  avatarDirection: AvatarDirection;
  previewDialogueState: DialogueState | null;
  dispatch: (event: EngineEvent) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setMuted: (muted: boolean) => void;
  setAvatarDirection: (direction: AvatarDirection) => void;
  previewState: (state: DialogueState | null) => void;
  reset: () => void;
  lastError: string | null;
}

const initialState = {
  connectionStatus: 'disconnected' as ConnectionStatus,
  dialogueState: 'idle' as DialogueState,
  amplitude: 0,
  muted: false,
  partialTranscript: '',
  finalTranscript: '',
  transcriptLanguage: null,
  response: '',
  responseLanguage: null,
  stages: [],
  doctor: null,
  avatarDirection: 'glassai' as AvatarDirection,
  previewDialogueState: null,
  lastError: null,
};

export const useJarvisStore = create<JarvisState>()(
  persist(
    (set) => ({
      ...initialState,
      dispatch: (event) => {
        switch (event.type) {
          case 'state_changed':
            set({
              dialogueState: event.state,
              ...(event.state === 'idle'
                ? {
                    partialTranscript: '',
                    finalTranscript: '',
                    response: '',
                  }
                : {}),
            });
            break;
          case 'partial_transcript':
            set({
              partialTranscript: event.text,
              transcriptLanguage: event.language,
            });
            break;
          case 'final_transcript':
            set({
              finalTranscript: event.text,
              transcriptLanguage: event.language,
              partialTranscript: '',
            });
            break;
          case 'response':
            set({
              response: event.text,
              responseLanguage: event.language,
            });
            break;
          case 'amplitude':
            set({ amplitude: event.level });
            break;
          case 'metrics':
            set({
              stages: event.stages,
              doctor: event.doctor,
            });
            break;
          case 'error':
            set({ lastError: event.message });
            break;
        }
      },
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      setMuted: (muted) => set({ muted }),
      setAvatarDirection: (avatarDirection) => set({ avatarDirection }),
      previewState: (previewDialogueState) => set({ previewDialogueState }),
      reset: () => set(initialState),
    }),
    {
      name: 'jarvis-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({ avatarDirection: state.avatarDirection }),
    },
  ),
);
