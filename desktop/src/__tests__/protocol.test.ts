import { describe, expect, it } from 'vitest';
import { STATE_COLORS, type DialogueState } from '../protocol';

const dialogueStates: DialogueState[] = ['idle', 'listening', 'processing', 'responding', 'confirming', 'executing', 'follow_up'];

describe('protocol', () => {
  it('defines colors for every dialogue state', () => {
    expect(Object.keys(STATE_COLORS).sort()).toEqual([...dialogueStates].sort());
  });

  it('uses valid hex color strings', () => {
    Object.values(STATE_COLORS).forEach((color) => {
      expect(color).toMatch(/^#[0-9A-F]{6}$/i);
    });
  });
});
