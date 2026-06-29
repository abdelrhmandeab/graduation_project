import { beforeEach, describe, expect, it } from 'vitest';
import { useJarvisStore } from '../stores/jarvisStore';

describe('jarvisStore', () => {
  beforeEach(() => {
    useJarvisStore.getState().reset();
  });

  it('updates dialogue state from state_changed events', () => {
    useJarvisStore.getState().dispatch({ type: 'state_changed', state: 'listening' });

    expect(useJarvisStore.getState().dialogueState).toBe('listening');
  });

  it('updates amplitude from amplitude events', () => {
    useJarvisStore.getState().dispatch({ type: 'amplitude', level: 0.42 });

    expect(useJarvisStore.getState().amplitude).toBe(0.42);
  });

  it('clears visible text when returning to idle', () => {
    useJarvisStore.setState({
      partialTranscript: 'open',
      finalTranscript: 'open the browser',
      response: 'Opening Chrome.',
    });

    useJarvisStore.getState().dispatch({ type: 'state_changed', state: 'idle' });

    expect(useJarvisStore.getState().partialTranscript).toBe('');
    expect(useJarvisStore.getState().finalTranscript).toBe('');
    expect(useJarvisStore.getState().response).toBe('');
  });

  it('updates partial transcript text and language', () => {
    useJarvisStore.getState().dispatch({ type: 'partial_transcript', text: 'open', language: 'en' });

    expect(useJarvisStore.getState().partialTranscript).toBe('open');
    expect(useJarvisStore.getState().transcriptLanguage).toBe('en');
  });

  it('sets final transcript and clears partial transcript', () => {
    useJarvisStore.setState({ partialTranscript: 'open' });

    useJarvisStore.getState().dispatch({ type: 'final_transcript', text: 'open the browser', language: 'en' });

    expect(useJarvisStore.getState().partialTranscript).toBe('');
    expect(useJarvisStore.getState().finalTranscript).toBe('open the browser');
    expect(useJarvisStore.getState().transcriptLanguage).toBe('en');
  });

  it('updates response text and language', () => {
    useJarvisStore.getState().dispatch({ type: 'response', text: 'Opening Chrome for you.', language: 'en' });

    expect(useJarvisStore.getState().response).toBe('Opening Chrome for you.');
    expect(useJarvisStore.getState().responseLanguage).toBe('en');
  });

  it('updates metrics stages and doctor checks', () => {
    const stages = [{ name: 'stt', duration_ms: 123 }];
    const doctor = { ok: true, checks: [{ name: 'mic', ok: true, details: 'ready' }] };

    useJarvisStore.getState().dispatch({ type: 'metrics', stages, doctor });

    expect(useJarvisStore.getState().stages).toEqual(stages);
    expect(useJarvisStore.getState().doctor).toEqual(doctor);
  });

  it('updates last error from error events', () => {
    useJarvisStore.getState().dispatch({ type: 'error', message: 'socket failed' });

    expect(useJarvisStore.getState().lastError).toBe('socket failed');
  });

  it('handles config and optimistic dashboard state updates', () => {
    useJarvisStore.getState().dispatch({
      type: 'config',
      values: {
        model: 'qwen3:4b',
        model_tier: 'auto',
        wake_mode: 'both',
        feature_flags: {
          NUMERIC_PARSING_ENABLED: true,
          AUTO_APP_DISCOVERY_ENABLED: true,
          MEDIA_DIRECT_DISPATCH_ENABLED: true,
          SYSTEM_VOLUME_CONTROL: true,
        },
        stt_backend: 'hybrid_elevenlabs',
        tts_backend: 'hybrid',
        persona: 'friendly',
      },
    });

    useJarvisStore.getState().setFeatureFlagLocal('NUMERIC_PARSING_ENABLED', false);
    useJarvisStore.getState().setConfigValueLocal('model', 'qwen3:8b');
    useJarvisStore.getState().setAppView('dashboard');

    expect(useJarvisStore.getState().config?.feature_flags.NUMERIC_PARSING_ENABLED).toBe(false);
    expect(useJarvisStore.getState().config?.model).toBe('qwen3:8b');
    expect(useJarvisStore.getState().appView).toBe('dashboard');
  });
});
