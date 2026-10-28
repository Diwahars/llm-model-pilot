"""Test script to demonstrate dynamic model registry."""

import os
from dotenv import load_dotenv

load_dotenv()

# Enable dynamic registry
os.environ["USE_DYNAMIC_REGISTRY"] = "true"

from model_registry import build_dynamic_registry

print("Building dynamic model registry from provider APIs...\n")

registry = build_dynamic_registry()

print(f"Total models found: {len(registry)}\n")
print("=" * 80)

# Group by provider
by_provider = {}
for key, config in registry.items():
    provider = config.provider_name
    if provider not in by_provider:
        by_provider[provider] = []
    by_provider[provider].append((key, config.model_id))

for provider, models in sorted(by_provider.items()):
    print(f"\n{provider.upper()} ({len(models)} models):")
    print("-" * 80)
    for key, model_id in models:
        print(f"  {key:50s} -> {model_id}")

print("\n" + "=" * 80)
print("\nTo use dynamic registry in the app, set USE_DYNAMIC_REGISTRY=true in your .env file")
