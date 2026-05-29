from __future__ import annotations

import transformers


if not hasattr(transformers, "HybridCache") and hasattr(transformers, "DynamicCache"):
    class HybridCache(transformers.DynamicCache):
        pass

    transformers.HybridCache = HybridCache
    if hasattr(transformers, "_objects"):
        transformers._objects["HybridCache"] = HybridCache
    if hasattr(transformers, "__all__") and "HybridCache" not in transformers.__all__:
        transformers.__all__.append("HybridCache")
