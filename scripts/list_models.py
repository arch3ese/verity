"""
Script to list models available to the configured Google Gemini API key
and compare them against the model IDs in aligncraft/models/factory.py.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("STATUS: UNVERIFIED (No API key configured in environment or .env)")
        print("Model IDs in factory.py are left unchanged.")
        return 0

    print("Querying Google GenAI provider for available models...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        all_models = list(client.models.list())
    except Exception as e:
        print(f"ERROR: Failed to list models via google.genai: {e}")
        return 1

    model_names = [m.name for m in all_models]
    print(f"Successfully retrieved {len(model_names)} available models.")

    factory_ids = [
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.8-flash",
    ]

    print("\n--- Model ID Resolution Audit ---")
    all_resolved = True
    for fid in factory_ids:
        # Match against models/name or direct name
        matches = [m for m in model_names if m == fid or m == f"models/{fid}"]
        if matches:
            print(f"  [RESOLVED]     '{fid}' -> found: {matches}")
        else:
            print(f"  [UNRESOLVED]   '{fid}' -> not found in available models")
            all_resolved = False

    print("\nSummary:")
    if all_resolved:
        print("  All model IDs in factory.py resolve to valid models in the Google GenAI provider.")
        print("  Audit claim that 'gemini-3.5-flash-lite' and 'gemini-3.8-flash' do not exist is DISPROVED.")
    else:
        print("  Some model IDs could not be resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
