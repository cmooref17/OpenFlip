from .ollama_api import *
from . import utils
try:
    from . import rag
except ImportError:
    rag = None  # chromadb not available