import { FormEvent, useState } from 'react';
import type { Language, UICommand } from '../../protocol';
import '../../styles/glow-search.css';

interface PromptInputProps {
  send: (cmd: UICommand) => void;
}

const ARABIC_TEXT_PATTERN = /[؀-ۿ]/;

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
    <form onSubmit={handleSubmit} className="glow-search min-w-0 flex-1">
      {/* rotating conic-gradient glow layers */}
      <div className="glow-layer glow-outer" aria-hidden="true" />
      <div className="glow-layer glow-mid" aria-hidden="true" />
      <div className="glow-layer glow-bright" aria-hidden="true" />
      <div className="glow-layer glow-border" aria-hidden="true" />

      <div className="relative flex w-full items-center">
        {/* left search icon (gradient stroke) */}
        <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2" aria-hidden="true">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="22"
            height="22"
            viewBox="0 0 24 24"
            fill="none"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="8" stroke="url(#glowSearchStroke)" />
            <line x1="22" y1="22" x2="16.65" y2="16.65" stroke="url(#glowSearchStrokeL)" />
            <defs>
              <linearGradient id="glowSearchStroke" gradientTransform="rotate(50)">
                <stop offset="0%" stopColor="#f8e7f8" />
                <stop offset="50%" stopColor="#b6a9b7" />
              </linearGradient>
              <linearGradient id="glowSearchStrokeL">
                <stop offset="0%" stopColor="#b6a9b7" />
                <stop offset="50%" stopColor="#837484" />
              </linearGradient>
            </defs>
          </svg>
        </span>

        <input
          type="text"
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type a prompt"
          className="h-14 w-full rounded-lg border-none bg-[#010201] pl-12 pr-14 text-base text-white outline-none placeholder:text-gray-400"
        />

        {/* right send button with spinning gradient border */}
        <button
          type="submit"
          aria-label="Send"
          title="Send"
          className="absolute right-2 top-1/2 flex h-10 w-10 -translate-y-1/2 cursor-pointer items-center justify-center overflow-hidden rounded-lg [isolation:isolate]"
        >
          <span className="icon-spin" aria-hidden="true" />
          <span className="flex h-full w-full items-center justify-center rounded-lg border border-transparent bg-gradient-to-b from-[#161329] via-black to-[#1d1b4b] text-[#d6d6e6] transition-colors hover:text-white">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M22 2 L11 13 M22 2 L15 22 L11 13 L2 9 Z"
                stroke="currentColor"
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
        </button>
      </div>
    </form>
  );
}
