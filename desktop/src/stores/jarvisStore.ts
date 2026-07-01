import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';
import type { ConfigValues, DialogueState, EngineEvent, FeatureFlags, Language } from '../protocol';

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected';
export type AvatarDirection = 'aurora' | 'glyph' | 'glassai' | 'companion';
export type AppView = 'overlay' | 'dashboard';
export type UiLanguage = Language | 'auto';

interface JarvisState {
  connectionStatus: ConnectionStatus;
  dialogueState: DialogueState;
  config: ConfigValues | null;
  appView: AppView;
  uiLanguage: UiLanguage;
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
  textPromptEnabled: boolean;
  dispatch: (event: EngineEvent) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setMuted: (muted: boolean) => void;
  setAvatarDirection: (direction: AvatarDirection) => void;
  setAppView: (view: AppView) => void;
  setUiLanguage: (language: UiLanguage) => void;
  setTextPromptEnabled: (enabled: boolean) => void;
  setFeatureFlagLocal: (flag: keyof FeatureFlags, enabled: boolean) => void;
  setConfigValueLocal: <K extends keyof ConfigValues>(key: K, value: ConfigValues[K]) => void;
  previewState: (state: DialogueState | null) => void;
  setPreviewState: (state: DialogueState | null) => void;
  reset: () => void;
  lastError: string | null;
}

const initialState = {
  connectionStatus: 'disconnected' as ConnectionStatus,
  dialogueState: 'idle' as DialogueState,
  config: null,
  appView: 'overlay' as AppView,
  uiLanguage: 'auto' as UiLanguage,
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
  textPromptEnabled: true,
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
          case 'config':
            set({ config: event.values });
            break;
        }
      },
      setConnectionStatus: (status) => set({ connectionStatus: status }),
      setMuted: (muted) => set({ muted }),
      setAvatarDirection: (avatarDirection) => set({ avatarDirection }),
      setAppView: (appView) => set({ appView }),
      setUiLanguage: (uiLanguage) => set({ uiLanguage }),
      setTextPromptEnabled: (textPromptEnabled) => set({ textPromptEnabled }),
      setFeatureFlagLocal: (flag, enabled) =>
        set((state) => {
          if (!state.config) {
            return state;
          }

          return {
            config: {
              ...state.config,
              feature_flags: {
                ...state.config.feature_flags,
                [flag]: enabled,
              },
            },
          };
        }),
      setConfigValueLocal: (key, value) =>
        set((state) => {
          if (!state.config) {
            return state;
          }

          return {
            config: {
              ...state.config,
              [key]: value,
            },
          };
        }),
      previewState: (previewDialogueState) => set({ previewDialogueState }),
      setPreviewState: (previewDialogueState) => set({ previewDialogueState }),
      reset: () => set(initialState),
    }),
    {
      name: 'jarvis-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        avatarDirection: state.avatarDirection,
        appView: state.appView,
        textPromptEnabled: state.textPromptEnabled,
        muted: state.muted,
      }),
    },
  ),
);

// Cross-window sync. Under Tauri the overlay and dashboard are separate OS
// windows, each with its own store instance. They share one localStorage
// (same origin), so when one window writes a persisted setting (avatar, view,
// text-prompt), the others rehydrate from it here. Also keeps browser tabs in
// sync. The storage event only fires in *other* documents, so no echo loop.
if (typeof window !== 'undefined') {
  window.addEventListener('storage', (event) => {
    if (event.key === 'jarvis-ui') {
      void useJarvisStore.persist.rehydrate();
    }
  });
}
