import os

# Avoid protobuf 4+/5+ incompatibility with older generated _pb2 modules (chromadb, etc.).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
