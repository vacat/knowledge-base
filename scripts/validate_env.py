#!/usr/bin/env python3
"""
Environment validator — checks that all required API keys are present.
Run this before ingest.py to fail fast on missing configuration.
"""
import os
import sys


def check(key: str, required: bool = True) -> bool:
    """Check if an environment variable is set."""
    present = bool(os.environ.get(key))
    status = "✓" if present else "✗"
    print(f"  [{status}] {key}")
    if required and not present:
        print(f"      → Missing! Set it in GitHub Secrets or .env")
    return present


def main():
    print("🔍 Validating environment variables...\n")

    print("LLM Provider (choose one):")
    has_openai = check("OPENAI_API_KEY", required=False)
    has_anthropic = check("ANTHROPIC_API_KEY", required=False)
    has_zhipu = check("ZHIPU_API_KEY", required=False)

    if not any([has_openai, has_anthropic, has_zhipu]):
        print("\n❌ No LLM API key found. At least one is required.")
        sys.exit(1)

    print("\nContent Extraction:")
    check("JINA_API_KEY", required=False)  # Optional — falls back to direct fetch
    check("BRAVE_API_KEY", required=False)  # Optional search API backup

    print("\nData Sources:")
    check("X_BEARER_TOKEN", required=False)  # At least one X token needed for bookmarks
    check("X_API_KEY", required=False)
    check("X_ACCESS_TOKEN", required=False)

    print("\nGitHub (always required in CI):")
    check("GH_PAT", required=False)
    check("GH_REPO", required=False)

    print("\n✅ Environment check complete.")
    print("   Note: Only required keys are enforced; optional keys enable features.")
    print("   In CI, X tokens are required for X bookmark ingestion.")


if __name__ == "__main__":
    main()
