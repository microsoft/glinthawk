GLINTHAWK_MODELS = {
    "llama2-70b": {
        "n_layers": 80,
    }
}

GLINTHAWK_MODELS_BLOBSTORE = "azure://glinthawk.blob.core.windows.net/models"
GLINTHAWK_PROMPT_BLOBSTORE = "azure://glinthawk.blob.core.windows.net/prompts"

GLINTHAWK_API_ROOT = "https://glinthawk.dev/api/"
GLINTHAWK_API_ACCESS_TOKEN = "REDACTED"

try:
    from settings_local import *
except ImportError:
    print("No local settings found")
    pass
