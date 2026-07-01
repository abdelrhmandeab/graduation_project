import { FormEvent, useState } from 'react';
import type { Language, UICommand } from '../../protocol';

interface PromptInputProps {
  send: (cmd: UICommand) => void;
}

const ARABIC_TEXT_PATTERN = /[\u0600-\u06FF]/;

function detectLanguage(text: string): Language {
  return ARABIC_TEXT_PATTERN.test(text) ? 'ar' : 'en';
}

export function PromptInput({ send }: PromptInputProps) {
  const [text, setText] = useState('');

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    const trimmedText = text.trim();
    if (!trimmedText) {
      return;
    }

    send({
      type: 'text_command',
      text: trimmedText,
      language: detectLanguage(trimmedText),
    });
    setText('');
  };

  return (
    <form onSubmit={handleSubmit} className="flex min-w-0 flex-1 items-center gap-2">
      <input
        type="text"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Type a prompt"
        className="h-10 min-w-0 flex-1 rounded border border-white/10 bg-white/[0.07] px-3 text-sm text-white outline-none transition focus:border-[#8EEBFF]/70 focus:bg-white/[0.1]"
      />
      <button
        type="submit"
        aria-label="Send"
        title="Send"
        className="grid h-10 w-10 shrink-0 place-items-center rounded border border-[#8EEBFF]/30 bg-[#8EEBFF]/12 text-[#DDFBFF] transition hover:bg-[#8EEBFF]/18"
      >
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M22 2 L11 13 M22 2 L15 22 L11 13 L2 9 Z"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </form>
  );
}
