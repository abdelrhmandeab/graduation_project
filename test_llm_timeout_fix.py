#!/usr/bin/env python3
"""Test to verify LLM timeout bug is fixed.

Reproduces the exact failing scenario:
1. Arabic weather query (successful)
2. English follow-up "How was it?" (was timing out after 45s with qwen3:4b)
3. Verify both queries complete within timeout with qwen2.5:3b
"""

import sys
import os
import time

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import LLM_TIMEOUT_SECONDS, LLM_MODEL
from llm.ollama_client import ask_llm_streaming, set_runtime_model, _resolve_model_name
from core.logger import logger

def test_scenario():
    """Test the exact scenario that was failing."""
    
    print(f"\n{'='*70}")
    print("LLM TIMEOUT FIX TEST")
    print(f"{'='*70}")
    print(f"Timeout configured: {LLM_TIMEOUT_SECONDS}s")
    print(f"Model configured: {LLM_MODEL}")
    
    # Set the runtime model explicitly
    set_runtime_model("qwen2.5:3b", num_ctx=4096)
    actual_model = _resolve_model_name()
    print(f"Runtime model: {actual_model}")
    print(f"{'='*70}\n")
    
    # Test 1: Arabic weather query (should work fast)
    print("[TEST 1] Arabic weather query (simulated as fast LLM_QUERY)")
    print("-" * 70)
    ar_prompt = """You are a helpful assistant. Answer briefly in Egyptian Arabic.

User asks: "عاوزك تقولي أخبار الطقس النهاردة في مصر"

Provide a weather response in Egyptian Arabic colloquial."""
    
    start = time.time()
    ar_response = ask_llm_streaming(ar_prompt, on_sentence=lambda x: print(f"  >> {x[:80]}", flush=True))
    ar_latency = time.time() - start
    print(f"✓ Arabic query: {ar_latency:.2f}s")
    print(f"  Response preview: {ar_response[:100]}...")
    print()
    
    # Test 2: English follow-up (THIS WAS TIMING OUT)
    print("[TEST 2] English follow-up query (the failing case)")
    print("-" * 70)
    en_prompt = """You are a helpful assistant. Keep responses concise.

Context: Last app used was notepad. The previous query was in Arabic about weather.

User asks: "How was it?"

Respond briefly in English, acknowledging the ambiguity."""
    
    start = time.time()
    en_response = ask_llm_streaming(en_prompt, on_sentence=lambda x: print(f"  >> {x[:80]}", flush=True))
    en_latency = time.time() - start
    print(f"✓ English query: {en_latency:.2f}s")
    print(f"  Response preview: {en_response[:100]}...")
    print()
    
    # Results
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Arabic query:  {ar_latency:.2f}s (target: <10s)")
    print(f"English query: {en_latency:.2f}s (target: <10s)")
    print(f"Total:         {ar_latency + en_latency:.2f}s")
    
    # Validation
    success = True
    if ar_latency > LLM_TIMEOUT_SECONDS:
        print(f"✗ Arabic query exceeded timeout! ({ar_latency:.2f}s > {LLM_TIMEOUT_SECONDS}s)")
        success = False
    else:
        print(f"✓ Arabic query within timeout")
    
    if en_latency > LLM_TIMEOUT_SECONDS:
        print(f"✗ English query exceeded timeout! ({en_latency:.2f}s > {LLM_TIMEOUT_SECONDS}s)")
        success = False
    else:
        print(f"✓ English query within timeout")
    
    # Check for thinking tag leakage
    if "We are given a user query" in en_response or "<think>" in en_response:
        print(f"⚠ WARNING: Response contains system thinking/prompts:")
        print(f"  {en_response[:150]}")
        success = False
    else:
        print(f"✓ No thinking tag leakage in response")
    
    print(f"{'='*70}")
    if success:
        print("✓ ALL TESTS PASSED - LLM TIMEOUT BUG IS FIXED!")
    else:
        print("✗ TESTS FAILED - Issue may not be fully resolved")
    print(f"{'='*70}\n")
    
    return success

if __name__ == "__main__":
    try:
        success = test_scenario()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
