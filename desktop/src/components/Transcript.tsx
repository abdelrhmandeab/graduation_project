import { AnimatePresence, motion } from 'motion/react';
import type { Language } from '../protocol';
import { useJarvisStore } from '../stores/jarvisStore';

function getDirection(language: Language | null) {
  return language === 'ar' ? 'rtl' : 'ltr';
}

export function Transcript() {
  const partialTranscript = useJarvisStore((state) => state.partialTranscript);
  const finalTranscript = useJarvisStore((state) => state.finalTranscript);
  const response = useJarvisStore((state) => state.response);
  const dialogueState = useJarvisStore((state) => state.dialogueState);
  const transcriptLanguage = useJarvisStore((state) => state.transcriptLanguage);
  const responseLanguage = useJarvisStore((state) => state.responseLanguage);

  let text = '';
  let language: Language | null = null;

  if (dialogueState === 'listening' && partialTranscript) {
    text = `${partialTranscript} ▍`;
    language = transcriptLanguage;
  } else if (dialogueState === 'processing' && finalTranscript) {
    text = finalTranscript;
    language = transcriptLanguage;
  } else if (dialogueState === 'responding' && response) {
    text = response;
    language = responseLanguage;
  }

  return (
    <AnimatePresence mode="wait">
      {text ? (
        <motion.div
          key={`${dialogueState}-${text}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          dir={getDirection(language)}
          className="max-w-[300px] text-center font-mono text-base text-white"
        >
          {text}
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
