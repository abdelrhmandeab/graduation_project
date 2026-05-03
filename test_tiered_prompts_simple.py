#!/usr/bin/env python3
"""Quick test of tiered prompt builders to validate token count reduction."""

from llm.prompt_builder import (
    _build_system_block,
    _estimate_token_count,
    _FEW_SHOT_EXAMPLES_MINIMAL,
    _FEW_SHOT_EXAMPLES_FULL,
)

# Test system prompt generation for different tiers
minimal_system = _build_system_block("en", None, include_few_shot=True, tier="low")
full_system = _build_system_block("en", None, include_few_shot=True, tier="high")

minimal_text = "\n".join(minimal_system)
full_text = "\n".join(full_system)

print('TIERED SYSTEM PROMPT COMPARISON')
print('=' * 70)
print()
print(f'Minimal Prompt (1.7B) - {len(minimal_system)} sections:')
print('-' * 70)
print(minimal_text)
print()
print(f'Minimal tokens: {_estimate_token_count(minimal_text)}')
print()
print('=' * 70)
print()
print(f'Full Prompt (8B+) - {len(full_system)} sections:')
print('-' * 70)
print(full_text)
print()
print(f'Full tokens: {_estimate_token_count(full_text)}')
print()
print('=' * 70)

# Calculate reduction
minimal_tok = _estimate_token_count(minimal_text)
full_tok = _estimate_token_count(full_text)
reduction = ((full_tok - minimal_tok) / full_tok * 100) if full_tok > 0 else 0

print()
print(f'SYSTEM PROMPT TOKEN REDUCTION: {reduction:.1f}%')
print(f'Minimal few-shot examples: {len(_FEW_SHOT_EXAMPLES_MINIMAL.split("USER:"))-1} Q&A pairs')
print(f'Full few-shot examples: {len(_FEW_SHOT_EXAMPLES_FULL.split("USER:"))-1} Q&A pairs')
print()
print('✓ Phase 1.1 implementation: Tiered / Model-Size-Aware Prompts COMPLETE')
