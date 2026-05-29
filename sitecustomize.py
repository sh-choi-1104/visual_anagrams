from __future__ import annotations

try:
    import transformers

    if not hasattr(transformers, "HybridCache") and hasattr(transformers, "DynamicCache"):
        class HybridCache(transformers.DynamicCache):
            pass

        transformers.HybridCache = HybridCache
except Exception:
    pass
