import type { ReactNode } from 'react';
import type { UICommand, FeatureFlags } from '../../protocol';
import { closeApp } from '../../lib/app';
import { useJarvisStore, type AvatarDirection, type UiLanguage } from '../../stores/jarvisStore';
import { Segmented, type SegmentedOption } from './Segmented';
import { Select, type SelectOption } from './Select';
import { Toggle } from './Toggle';

interface DashboardProps {
  send: (cmd: UICommand) => void;
}

const avatarOptions: Array<SegmentedOption<AvatarDirection>> = [
  { label: 'Aurora', value: 'aurora' },
  { label: 'Glyph', value: 'glyph' },
  { label: 'Glass AI', value: 'glassai' },
  { label: 'Companion', value: 'companion' },
];

const languageOptions: Array<SegmentedOption<UiLanguage>> = [
  { label: 'English', value: 'en' },
  { label: 'Arabic', value: 'ar' },
  { label: 'Auto', value: 'auto' },
];

const personaOptions: SelectOption[] = [
  { label: 'Friendly', value: 'friendly' },
  { label: 'Formal', value: 'formal' },
  { label: 'Casual', value: 'casual' },
  { label: 'Professional', value: 'professional' },
  { label: 'Brief', value: 'brief' },
];

const modelOptions: SelectOption[] = [
  { label: 'Auto', value: 'auto' },
  { label: 'qwen3:0.6b', value: 'qwen3:0.6b' },
  { label: 'qwen3:1.7b', value: 'qwen3:1.7b' },
  { label: 'qwen3:4b', value: 'qwen3:4b' },
  { label: 'qwen3:8b', value: 'qwen3:8b' },
];

const featureFlagLabels: Record<keyof FeatureFlags, string> = {
  NUMERIC_PARSING_ENABLED: 'Numeric parsing',
  AUTO_APP_DISCOVERY_ENABLED: 'Auto app discovery',
  MEDIA_DIRECT_DISPATCH_ENABLED: 'Media direct dispatch',
  SYSTEM_VOLUME_CONTROL: 'System volume control',
};

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded border border-white/10 bg-white/[0.045] p-4 shadow-lg shadow-black/20">
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-[#8EEBFF]/78">{title}</h2>
      <div className="grid gap-3">{children}</div>
    </section>
  );
}

export function Dashboard({ send }: DashboardProps) {
  const config = useJarvisStore((state) => state.config);
  const avatarDirection = useJarvisStore((state) => state.avatarDirection);
  const uiLanguage = useJarvisStore((state) => state.uiLanguage);
  const muted = useJarvisStore((state) => state.muted);
  const connectionStatus = useJarvisStore((state) => state.connectionStatus);
  const setAppView = useJarvisStore((state) => state.setAppView);
  const setAvatarDirection = useJarvisStore((state) => state.setAvatarDirection);
  const setUiLanguage = useJarvisStore((state) => state.setUiLanguage);
  const setMuted = useJarvisStore((state) => state.setMuted);
  const setFeatureFlagLocal = useJarvisStore((state) => state.setFeatureFlagLocal);
  const setConfigValueLocal = useJarvisStore((state) => state.setConfigValueLocal);

  const hasConfig = config !== null;

  const handleLanguageChange = (language: UiLanguage) => {
    setUiLanguage(language);
    send({ type: 'setting_update', key: 'JARVIS_STT_LANGUAGE_HINT', value: language });
  };

  const handleMutedChange = (nextMuted: boolean) => {
    setMuted(nextMuted);
    send({ type: 'mute_toggle', muted: nextMuted });
  };

  return (
    <div className="min-h-screen overflow-y-auto bg-[#0A0A0F] px-4 py-6 text-white sm:px-6 lg:px-8">
      <main className="mx-auto grid w-full max-w-5xl gap-5 rounded border border-white/10 bg-white/[0.035] p-4 shadow-2xl shadow-black/45 backdrop-blur-2xl sm:p-6">
        <header className="flex flex-col gap-3 border-b border-white/10 pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-[#8EEBFF]/70">Control Center</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-normal text-white">Jarvis Dashboard</h1>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => send({ type: 'config_request' })}
              className="h-10 rounded border border-white/10 bg-white/[0.06] px-4 text-sm font-medium text-white/78 transition hover:border-[#8EEBFF]/35 hover:text-white"
            >
              Refresh
            </button>
            <button
              type="button"
              onClick={() => setAppView('overlay')}
              className="h-10 rounded border border-[#8EEBFF]/28 bg-[#8EEBFF]/12 px-4 text-sm font-medium text-[#DDFBFF] transition hover:bg-[#8EEBFF]/18"
            >
              Back
            </button>
          </div>
        </header>

        {!hasConfig ? (
          <div className="rounded border border-[#8EEBFF]/18 bg-[#8EEBFF]/8 p-4 text-sm text-white/72">
            Engine config has not arrived yet. Use Refresh to request the current values from the bridge.
          </div>
        ) : null}

        <div className="grid gap-4 lg:grid-cols-2">
          <Section title="Avatar">
            <Segmented value={avatarDirection} options={avatarOptions} onChange={setAvatarDirection} />
          </Section>

          <Section title="Voice Persona">
            <Select
              label="Persona"
              value={config?.persona ?? 'friendly'}
              options={personaOptions}
              disabled={!hasConfig}
              onChange={(persona) => {
                setConfigValueLocal('persona', persona);
                send({ type: 'setting_update', key: 'JARVIS_PERSONA', value: persona });
              }}
            />
          </Section>

          <Section title="Language">
            <Segmented value={uiLanguage} options={languageOptions} onChange={handleLanguageChange} />
          </Section>

          <Section title="Model">
            <Select
              label="LLM model"
              value={config?.model ?? 'auto'}
              options={modelOptions}
              disabled={!hasConfig}
              onChange={(model) => {
                setConfigValueLocal('model', model);
                send({ type: 'setting_update', key: 'JARVIS_LLM_MODEL', value: model });
              }}
            />
          </Section>

          <Section title="Feature Flags">
            {(Object.keys(featureFlagLabels) as Array<keyof FeatureFlags>).map((flag) => (
              <Toggle
                key={flag}
                label={featureFlagLabels[flag]}
                checked={config?.feature_flags[flag] ?? false}
                disabled={!hasConfig}
                onChange={(enabled) => {
                  setFeatureFlagLocal(flag, enabled);
                  send({ type: 'feature_flag', flag, enabled });
                }}
              />
            ))}
          </Section>

          <Section title="Audio">
            <Toggle label="Mute microphone and speech" checked={muted} onChange={handleMutedChange} />
          </Section>

          <Section title="Status">
            <dl className="grid gap-3 text-sm">
              <div className="flex items-center justify-between gap-4">
                <dt className="text-white/58">Connection</dt>
                <dd className="font-medium capitalize text-white/88">{connectionStatus}</dd>
              </div>
              <div className="flex items-center justify-between gap-4">
                <dt className="text-white/58">Current model</dt>
                <dd className="font-medium text-white/88">{config?.model ?? 'Unknown'}</dd>
              </div>
            </dl>
          </Section>

          <section className="rounded border border-red-300/20 bg-red-500/[0.055] p-4 lg:col-span-2">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-[0.14em] text-red-100/70">Close App</h2>
            <button
              type="button"
              onClick={() => {
                void closeApp().catch((error: unknown) => console.error('Failed to close app.', error));
              }}
              className="h-11 rounded border border-red-200/25 bg-red-400/14 px-4 text-sm font-semibold text-red-50 transition hover:bg-red-400/20"
            >
              Close Jarvis
            </button>
          </section>
        </div>
      </main>
    </div>
  );
}
