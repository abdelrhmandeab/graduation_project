export type DialogueState = 'idle' | 'listening' | 'processing' | 'responding' | 'confirming' | 'executing' | 'follow_up';

export type Language = 'en' | 'ar';

export interface FeatureFlags {
  NUMERIC_PARSING_ENABLED: boolean;
  AUTO_APP_DISCOVERY_ENABLED: boolean;
  MEDIA_DIRECT_DISPATCH_ENABLED: boolean;
  SYSTEM_VOLUME_CONTROL: boolean;
}

export interface ConfigValues {
  model: string;
  model_tier: string;
  wake_mode: string;
  feature_flags: FeatureFlags;
  stt_backend: string;
  tts_backend: string;
  persona: string;
}

// Engine -> UI events (discriminated union on "type" field)
export interface StateChangedEvent {
  type: 'state_changed';
  state: DialogueState;
}
export interface PartialTranscriptEvent {
  type: 'partial_transcript';
  text: string;
  language: Language;
}
export interface FinalTranscriptEvent {
  type: 'final_transcript';
  text: string;
  language: Language;
}
export interface ResponseEvent {
  type: 'response';
  text: string;
  language: Language;
}
export interface AmplitudeEvent {
  type: 'amplitude';
  level: number; // 0.0 - 1.0
}
export interface MetricsEvent {
  type: 'metrics';
  stages: Array<{ name: string; duration_ms: number }>;
  doctor: {
    ok: boolean;
    checks: Array<{ name: string; ok: boolean; details: string }>;
  };
}
export interface ErrorEvent {
  type: 'error';
  message: string;
}
export interface ConfigEvent {
  type: 'config';
  values: ConfigValues;
}

export type EngineEvent =
  | StateChangedEvent
  | PartialTranscriptEvent
  | FinalTranscriptEvent
  | ResponseEvent
  | AmplitudeEvent
  | MetricsEvent
  | ErrorEvent
  | ConfigEvent;

// UI -> Engine commands
export interface TextCommandMessage {
  type: 'text_command';
  text: string;
  language?: Language;
}
export interface MuteToggleMessage {
  type: 'mute_toggle';
  muted: boolean;
}
export interface SettingUpdateMessage {
  type: 'setting_update';
  key: string;
  value: unknown;
}
export interface FeatureFlagMessage {
  type: 'feature_flag';
  flag: keyof FeatureFlags;
  enabled: boolean;
}
export interface ConfigRequestMessage {
  type: 'config_request';
}

export type UICommand =
  | TextCommandMessage
  | MuteToggleMessage
  | SettingUpdateMessage
  | FeatureFlagMessage
  | ConfigRequestMessage;

// State colors matching Python ui/tray.py
export const STATE_COLORS: Record<DialogueState, string> = {
  idle: '#5A5A5A',
  listening: '#007E00',
  processing: '#B28C00',
  responding: '#0054B2',
  confirming: '#B26200',
  executing: '#3F3F8C',
  follow_up: '#007054',
} as const;

export const MOCK_WS_PORT = 8765;

// WebSocket URL the UI connects to.
// Defaults to the in-process Vite mock server (port 8765, no path).
// Set VITE_JARVIS_WS_URL to target the real Python bridge, e.g.
//   VITE_JARVIS_WS_URL=ws://127.0.0.1:9720/ws
// `import.meta.env` is undefined when this module is evaluated in a plain Node
// context (e.g. vite.config.ts loading the mock server), so guard the access.
const viteEnv = import.meta.env as ImportMetaEnv | undefined;
export const JARVIS_WS_URL =
  viteEnv?.VITE_JARVIS_WS_URL ?? `ws://localhost:${MOCK_WS_PORT}`;
