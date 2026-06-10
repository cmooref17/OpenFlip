import os
import hashlib
import asyncio
from typing import Callable
from .utils import print_ts, COLOR_YELLOW, COLOR_END
from . import ollama_api as _ollama_api_module

try:
    import chromadb as _chromadb
    from chromadb.utils import embedding_functions as _embedding_functions
    _chromadb_available = True
except ImportError:
    _chromadb = None
    _embedding_functions = None
    _chromadb_available = False


def _rag_config() -> dict:
    """Read RAG config dynamically so set_host/load_config changes are picked up."""
    return _ollama_api_module.config.get('rag', {}) or {}


def _ollama_host() -> str:
    return _ollama_api_module.config.get('host', 'http://localhost:11434')


def _resolve_db_path(path: str) -> str:
    """Resolve vector_db_path: absolute paths are kept, relative paths are placed next to this module."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(os.path.dirname(__file__), path))


def chunk_text(text:str, chunk_size:int = None, overlap:int = None) -> list[str]:
    """Split text into overlapping chunks."""
    cfg = _rag_config()
    chunk_size = chunk_size if chunk_size is not None else cfg.get('chunk_size', 512)
    overlap = overlap if overlap is not None else cfg.get('chunk_overlap', 50)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, {chunk_size}); got {overlap}")
    if not text:
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


class Document:
    """Document chunk with metadata."""
    def __init__(self, content:str, source:str = None, metadata:dict = None, doc_id:str = None):
        self.content = content
        self.source = source or "unknown"
        self.metadata = metadata or {}
        if doc_id:
            self.id = doc_id
        else:
            # Hash full content + source + chunk_index (if present) so similar prefixes don't collide.
            chunk_index = self.metadata.get('chunk_index', '')
            payload = f"{self.source}\x00{chunk_index}\x00{self.content}".encode('utf-8')
            self.id = hashlib.md5(payload).hexdigest()


class RAGResult:
    """Retrieved context result."""
    def __init__(self, documents:list[Document] = None, query:str = None, scores:list[float] = None):
        self.documents = documents or []
        self.query = query or ""
        self.scores = scores or []

    def format_context(self, format_string:str = None) -> str:
        """Format retrieved documents as context string."""
        if not self.documents:
            return ""
        format_string = format_string or _rag_config().get('context_format', "[Context from {source}]\n{content}\n---")
        formatted = []
        for doc in self.documents:
            formatted.append(format_string.format(source=doc.source, content=doc.content))
        return "\n".join(formatted)


class RAGCollection:
    """Vector database collection wrapper."""
    _client = None
    _client_path: str | None = None

    @classmethod
    def _get_client(cls):
        if not _chromadb_available:
            raise ImportError("chromadb is not installed. Install it with: pip install chromadb")
        db_path = _resolve_db_path(_rag_config().get('vector_db_path', './data/chromadb'))
        # Rebuild the singleton if the path changed (config reload).
        if cls._client is None or cls._client_path != db_path:
            os.makedirs(db_path, exist_ok=True)
            cls._client = _chromadb.PersistentClient(path=db_path)
            cls._client_path = db_path
        return cls._client

    def __init__(self, collection_name:str = "default"):
        self.collection_name = collection_name
        self._collection = None
        self._embedding_fn = None
        self._embedding_signature = None  # (model, host) — rebuild fn on change

    def _get_embedding_function(self):
        cfg = _rag_config()
        embedding_model = cfg.get('embedding_model', 'nomic-embed-text')
        host = _ollama_host()
        signature = (embedding_model, host)
        if self._embedding_fn is None or self._embedding_signature != signature:
            self._embedding_fn = _embedding_functions.OllamaEmbeddingFunction(
                model_name=embedding_model,
                url=host,
            )
            self._embedding_signature = signature
            self._collection = None  # collection is bound to embedding fn — must rebuild
        return self._embedding_fn

    def _get_collection(self):
        embedding_fn = self._get_embedding_function()
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_fn,
            )
        return self._collection

    @property
    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def add(self, documents:list[Document], batch_size:int = 100) -> int:
        """Add documents to the collection. Returns number actually written."""
        if not documents:
            return 0
        try:
            collection = self._get_collection()

            # Dedup within the incoming batch.
            seen_ids = set()
            unique_docs = []
            for doc in documents:
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    unique_docs.append(doc)
            if not unique_docs:
                return 0

            ids = [d.id for d in unique_docs]
            contents = [d.content for d in unique_docs]
            metadatas = [{"source": d.source, **d.metadata} for d in unique_docs]

            # Find which ids already exist server-side.
            existing_ids = set()
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                existing = collection.get(ids=batch_ids)
                if existing and existing.get('ids'):
                    existing_ids.update(existing['ids'])

            new_ids, new_contents, new_metadatas = [], [], []
            for i, doc_id in enumerate(ids):
                if doc_id not in existing_ids:
                    new_ids.append(doc_id)
                    new_contents.append(contents[i])
                    new_metadatas.append(metadatas[i])
            if not new_ids:
                return 0

            total_added = 0
            for i in range(0, len(new_ids), batch_size):
                collection.add(
                    ids=new_ids[i:i + batch_size],
                    documents=new_contents[i:i + batch_size],
                    metadatas=new_metadatas[i:i + batch_size],
                )
                total_added += min(batch_size, len(new_ids) - i)
            return total_added
        except Exception as e:
            print_ts(f"RAG add error: {e}", error=True)
            return 0

    def query(self, query_text:str, top_k:int = None, threshold:float = None) -> RAGResult:
        """Query for similar documents. When threshold > 0, oversamples to try to return up to top_k passing results."""
        cfg = _rag_config()
        top_k = top_k if top_k is not None else cfg.get('retrieval_top_k', 5)
        threshold = threshold if threshold is not None else cfg.get('similarity_threshold', 0.0)
        # Oversample so threshold filtering doesn't shrink the result below top_k unnecessarily.
        fetch_k = top_k * 4 if threshold > 0 else top_k
        try:
            collection = self._get_collection()
            results = collection.query(
                query_texts=[query_text],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"],
            )

            documents, scores = [], []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc_text in enumerate(results['documents'][0]):
                    distance = results['distances'][0][i] if results.get('distances') else 0
                    score = 1 - distance
                    if score < threshold:
                        continue
                    metadata = dict(results['metadatas'][0][i]) if results.get('metadatas') else {}
                    source = metadata.pop('source', 'unknown')
                    documents.append(Document(doc_text, source, metadata))
                    scores.append(score)
                    if len(documents) >= top_k:
                        break
            return RAGResult(documents, query_text, scores)
        except Exception as e:
            print_ts(f"RAG query error: {e}", error=True)
            return RAGResult([], query_text, [])

    def delete(self, document_ids:list[str] = None, where:dict = None) -> bool:
        """Delete documents by ID or filter."""
        try:
            collection = self._get_collection()
            if document_ids:
                collection.delete(ids=document_ids)
            elif where:
                collection.delete(where=where)
            return True
        except Exception as e:
            print_ts(f"RAG delete error: {e}", error=True)
            return False

    def delete_by_source(self, source:str) -> int:
        """Delete all chunks tagged with the given source. Returns the number deleted (or -1 on error)."""
        try:
            collection = self._get_collection()
            existing = collection.get(where={"source": source})
            ids = (existing or {}).get('ids') or []
            if not ids:
                return 0
            collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            print_ts(f"RAG delete_by_source error: {e}", error=True)
            return -1

    def clear(self) -> bool:
        """Drop the entire collection."""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            self._collection = None
            return True
        except Exception as e:
            print_ts(f"RAG clear error: {e}", error=True)
            return False


# Process-wide collection cache so repeat calls reuse client + embedding fn.
_collections: dict[str, RAGCollection] = {}


def get_collection(collection_name: str = "default") -> RAGCollection:
    """Get (or create) a cached RAGCollection."""
    coll = _collections.get(collection_name)
    if coll is None:
        coll = RAGCollection(collection_name)
        _collections[collection_name] = coll
    return coll


async def retrieve(query:str, collection_name:str = "default", top_k:int = None, threshold:float = None) -> RAGResult:
    """Retrieve relevant documents for a query."""
    if not _rag_config().get('enable'):
        return RAGResult([], query, [])
    if not _chromadb_available:
        print_ts(f"{COLOR_YELLOW}RAG retrieve called but chromadb is not installed.{COLOR_END}")
        return RAGResult([], query, [])

    collection = get_collection(collection_name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, collection.query, query, top_k, threshold)


async def store(documents:list[Document], collection_name:str = "default") -> int:
    """Store documents in vector database. Returns number of documents added."""
    if not documents:
        return 0
    if not _chromadb_available:
        print_ts(f"{COLOR_YELLOW}RAG store called but chromadb is not installed.{COLOR_END}")
        return 0

    collection = get_collection(collection_name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, collection.add, documents)


async def delete_by_source(source:str, collection_name:str = "default") -> int:
    """Delete all chunks for a given source. Returns count deleted (or -1 on error)."""
    if not _chromadb_available:
        return 0
    collection = get_collection(collection_name)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, collection.delete_by_source, source)


async def store_text(text:str, source:str = "unknown", collection_name:str = "default", chunk_size:int = None, overlap:int = None, replace:bool = False) -> int:
    """Chunk and store text. If replace=True, deletes any prior chunks for this source first."""
    if replace:
        await delete_by_source(source, collection_name)
    chunks = chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0
    documents = [Document(chunk, source, {"chunk_index": i}) for i, chunk in enumerate(chunks)]
    return await store(documents, collection_name)


async def store_file(filepath:str, collection_name:str = "default", chunk_size:int = None, overlap:int = None, replace:bool = False, source:str = None) -> int:
    """Store a text file. If replace=True, removes prior chunks for the same source first."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print_ts(f"Failed to read file \"{filepath}\": {e}", error=True)
        return 0

    source = source or os.path.basename(filepath)
    return await store_text(content, source, collection_name, chunk_size, overlap, replace=replace)


async def store_directory(dirpath:str, collection_name:str = "default", extensions:list[str] = None, chunk_size:int = None, overlap:int = None, recursive:bool = True, replace:bool = False, progress:Callable[[str, int, int], None] = None) -> int:
    """Store all matching files from a directory. Recursive by default.

    progress(filepath, chunks_added, total_files_processed) is called after each file when supplied.
    """
    extensions = extensions or [".txt", ".md"]
    total = 0
    files_processed = 0
    try:
        if recursive:
            walker = ((root, fname) for root, _dirs, files in os.walk(dirpath) for fname in files)
        else:
            walker = ((dirpath, fname) for fname in os.listdir(dirpath))
        for root, filename in walker:
            if not any(filename.endswith(ext) for ext in extensions):
                continue
            filepath = os.path.join(root, filename)
            if not os.path.isfile(filepath):
                continue
            count = await store_file(filepath, collection_name, chunk_size, overlap, replace=replace)
            if count > 0:
                print_ts(f"Stored {count} chunks from {os.path.relpath(filepath, dirpath)}")
            total += count
            files_processed += 1
            if progress:
                try:
                    progress(filepath, count, files_processed)
                except Exception as e:
                    print_ts(f"RAG progress callback error: {e}", error=True)
    except Exception as e:
        print_ts(f"Failed to read directory \"{dirpath}\": {e}", error=True)
    return total


def is_available() -> bool:
    """Check if RAG is available (chromadb installed and enabled)."""
    return _chromadb_available and bool(_rag_config().get('enable', False))


if __name__ == '__main__':
    pass
