#!/usr/bin/env python3
"""Quick test of tiered prompt builders to validate token count reduction."""

from llm.prompt_builder import build_minimal_prompt, build_full_prompt, build_prompt_package

test_query = 'what is the weather?'

# Get minimal prompt
minimal = build_minimal_prompt(test_query)
minimal_tokens = minimal['token_count']
minimal_lines = len(minimal['prompt'].split('\n'))

# Get full prompt
full = build_full_prompt(test_query)
full_tokens = full['token_count']
full_lines = len(full['prompt'].split('\n'))

# Get medium (default)
medium = build_prompt_package(test_query, tier='medium')
medium_tokens = medium['token_count']
medium_lines = len(medium['prompt'].split('\n'))

print('TIERED PROMPT COMPARISON')
print('=' * 60)
print(f'Minimal (1.7B):  {minimal_tokens:4d} tokens, {minimal_lines:2d} lines (tier={minimal["tier"]})')
print(f'Medium (4B):     {medium_tokens:4d} tokens, {medium_lines:2d} lines (tier={medium["tier"]})')
print(f'Full (8B+):      {full_tokens:4d} tokens, {full_lines:2d} lines (tier={full["tier"]})')
print('=' * 60)
reduction = ((full_tokens - minimal_tokens) / full_tokens) * 100 if full_tokens > 0 else 0
print(f'Minimal vs Full: {reduction:.1f}% token reduction')
print(f'Few-shot examples: Minimal has 2-3, Full has 4+ entries')
print()
print('MINIMAL PROMPT SAMPLE:')
print('-' * 60)
print(minimal['prompt'][:300] + '...')
print()
print('FULL PROMPT SAMPLE:')
print('-' * 60)
print(full['prompt'][:300] + '...')
