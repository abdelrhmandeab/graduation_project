import { FormEvent, useState } from 'react';
import type { Language, UICommand } from '../../protocol';

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
    <form onSubmit={handleSubmit} className="flex min-w-0 flex-1 items-center gap-2">
      <input
        type="text"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Type a prompt"
        className="min-w-0 flex-1 bg-transparent px-2 py-2 font-sans text-sm font-medium text-gray-900 outline-none placeholder:text-gray-500 dark:text-white dark:placeholder:text-white/40"
      />
      {/* Gradient inset submit button (design style) */}
      <button
        type="submit"
        aria-label="Send"
        title="Send"
        className="group flex shrink-0 cursor-pointer rounded-lg bg-gradient-to-t from-gray-400 via-gray-300 to-gray-500 p-1 shadow-inner outline-none transition-all duration-150 active:scale-95 dark:from-gray-800 dark:via-gray-600 dark:to-gray-800"
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/20 text-gray-600 backdrop-blur-sm transition-all duration-300 group-hover:text-gray-900 group-hover:drop-shadow-[0_0_4px_rgba(0,0,0,0.35)] dark:bg-black/10 dark:text-gray-300 dark:group-hover:text-white dark:group-hover:drop-shadow-[0_0_5px_#fff]">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
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
    </form>
  );
}
