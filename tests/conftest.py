import os

# Tests must never export telemetry using credentials from a developer's .env file.
os.environ["ENABLE_LANGFUSE"] = "false"
os.environ["PRELOAD_RAG_SERVICE"] = "false"
