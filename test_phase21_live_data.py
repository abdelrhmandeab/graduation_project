"""Phase 2.1 Live Data Pipeline Smoke Tests"""
import time
from tools.live_data import gather_live_data

# Test cases
test_queries = [
    "What is the weather?",
    "Tell me the current weather in Cairo",
    "Search for python machine learning",
    "Look up the latest news about AI",
    "How is the weather today and what should I wear?",
    "طقس الاسكندرية",
    "ابحث عن أخبار تكنولوجيا",
]

print("Phase 2.1: Live Data Pipeline Tests\n")
print("=" * 70)

for query in test_queries:
    start = time.perf_counter()
    try:
        result = gather_live_data(query, parallel=True)
        elapsed = time.perf_counter() - start
        
        if result:
            preview = result.replace("\n", " ")[:60] + "..." if len(result) > 60 else result
            print(f"✓ [{elapsed*1000:6.1f}ms] {query[:40]:<40} → {preview}")
        else:
            print(f"- [{elapsed*1000:6.1f}ms] {query[:40]:<40} → No live data")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"✗ [{elapsed*1000:6.1f}ms] {query[:40]:<40} → ERROR: {str(e)[:40]}")

print("=" * 70)
print("Notes:")
print("- Latency target: <1s cached, ~2-3s fresh")
print("- First run may be slower due to network fetch")
print("- Parallel execution should keep total time low")
