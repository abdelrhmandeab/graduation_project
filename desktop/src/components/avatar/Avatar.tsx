import { STATE_COLORS } from '../../protocol';
import { useJarvisStore } from '../../stores/jarvisStore';
import { AuroraAvatar } from './AuroraAvatar';
import { CompanionAvatar } from './CompanionAvatar';
import { GlassAIAvatar } from './GlassAIAvatar';
import { GlyphAvatar } from './GlyphAvatar';

export function Avatar() {
  const avatarDirection = useJarvisStore((state) => state.avatarDirection);
  const dialogueState = useJarvisStore((state) => state.previewDialogueState ?? state.dialogueState);
  const amplitude = useJarvisStore((state) => state.amplitude);
  const color = STATE_COLORS[dialogueState];

  switch (avatarDirection) {
    case 'aurora':
      return <AuroraAvatar state={dialogueState} amplitude={amplitude} color={color} />;
    case 'glyph':
      return <GlyphAvatar state={dialogueState} amplitude={amplitude} color={color} />;
    case 'companion':
      return <CompanionAvatar state={dialogueState} color={color} />;
    case 'glassai':
    default:
      return <GlassAIAvatar state={dialogueState} color={color} />;
  }
}
