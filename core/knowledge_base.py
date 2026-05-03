import json
import os
import re
import threading
import time
from hashlib import sha1, sha256

import numpy as np

from core.config import (
    KB_BLOCKED_CONTEXT_PATTERNS,
    KB_AUTO_SYNC_ENABLED,
    KB_AUTO_SYNC_INTERVAL_SECONDS,
    KB_AUTO_SYNC_PATHS,
    KB_CHUNK_OVERLAP,
    KB_CHUNK_SIZE,
    KB_EMBEDDING_DIM,
    KB_EMBEDDING_MODEL,
    KB_EMBEDDING_RERANK_WEIGHT,
    KB_ENABLED,
    KB_FAISS_INDEX_FILE,
    KB_LEXICAL_RERANK_WEIGHT,
    KB_MAX_CONTEXT_CHARS,
    KB_MIN_PROMPT_SCORE,
    KB_MIN_SEMANTIC_ONLY_SCORE,
    KB_META_FILE,
    KB_RETRIEVAL_ENABLED,
    KB_RERANK_CANDIDATE_MULTIPLIER,
    KB_SOURCE_STATE_FILE,
    KB_STORAGE_DIR,
    KB_TOP_K,
)
from core.logger import logger

try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    _WATCHDOG_AVAILABLE = True
except Exception:
    Observer = None
    FileSystemEventHandler = object
    _WATCHDOG_AVAILABLE = False

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
_SUSPICIOUS_PREFIX_RE = re.compile(r"^\s*(system|assistant|developer)\s*:\s*", flags=re.IGNORECASE)


def _safe_read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
        return file_handle.read()


def _safe_read_pdf(path):
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return ""

    text_parts = []
    try:
        reader = PdfReader(path)
        for page in reader.pages:
            text_parts.append((page.extract_text() or "").strip())
    except Exception:
        return ""
    return "\n".join(part for part in text_parts if part)


def _is_supported_knowledge_file(filename):
    lower = filename.lower()
    return lower.endswith((".txt", ".md", ".rst", ".py", ".json", ".csv", ".pdf"))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


class EmbeddingBackend:
    def __init__(self):
        self._model = None
        self._mode = "hash"
        self._dim = int(KB_EMBEDDING_DIM)
        self._lock = threading.Lock()

    def _ensure_model(self):
        if SentenceTransformer is None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                model = SentenceTransformer(KB_EMBEDDING_MODEL)
                sample = model.encode(["ping"], normalize_embeddings=True)
                dim = int(sample.shape[1])
                self._model = model
                self._mode = "sentence_transformers"
                self._dim = dim
                logger.info("Knowledge base embeddings backend: sentence_transformers (%s)", KB_EMBEDDING_MODEL)
            except Exception as exc:
                logger.warning("Knowledge embeddings fallback to hash backend: %s", exc)

    @property
    def mode(self):
        self._ensure_model()
        return self._mode

    @property
    def dimension(self):
        self._ensure_model()
        return self._dim

    def embed(self, texts):
        self._ensure_model()
        if self._model is not None:
            vectors = self._model.encode(list(texts), normalize_embeddings=True)
            return np.asarray(vectors, dtype=np.float32)
        return self._hash_embed_many(texts)

    def _hash_embed_many(self, texts):
        dim = self._dim
        output = np.zeros((len(texts), dim), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = _TOKEN_RE.findall((text or "").lower())
            if not tokens:
                continue

            for token in tokens:
                digest = sha1(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % dim
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                output[row_index, bucket] += sign

            norm = float(np.linalg.norm(output[row_index]))
            if norm > 0:
                output[row_index] /= norm
        return output


class KnowledgeBaseService:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = bool(KB_ENABLED)
        self._retrieval_enabled = bool(KB_RETRIEVAL_ENABLED)
        self._embedder = EmbeddingBackend()
        self._meta = []
        self._vectors = np.zeros((0, KB_EMBEDDING_DIM), dtype=np.float32)
        self._faiss_index = None
        self._source_state = {}
        self._vectors_file = os.path.join(KB_STORAGE_DIR, "vectors.npy")
        self._initialized = False
        self._auto_sync_enabled = bool(KB_AUTO_SYNC_ENABLED)
        self._auto_sync_interval_seconds = max(1.0, float(KB_AUTO_SYNC_INTERVAL_SECONDS or 4.0))
        self._auto_sync_roots = [
            os.path.abspath(os.path.expanduser(path))
            for path in (KB_AUTO_SYNC_PATHS or [])
            if str(path or "").strip()
        ]
        self._auto_sync_thread = None
        self._auto_sync_stop_event = threading.Event()
        self._auto_sync_last_signature = ""
        self._auto_sync_last_run_ts = 0.0
        self._auto_sync_last_error = ""
        self._auto_sync_last_changes = {
            "indexed_files": 0,
            "skipped_files": 0,
            "removed_files": 0,
        }
        # Watchdog-related fields (optional, fallback to polling loop)
        self._watchdog_observer = None
        self._watchdog_handlers = []
        self._watchdog_debounce_timer = None
        self._watchdog_debounce_lock = threading.Lock()

    def _ensure_initialized(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            os.makedirs(KB_STORAGE_DIR, exist_ok=True)
            self._load_from_disk()
            self._initialized = True
            if self._auto_sync_enabled and self._auto_sync_roots:
                self.start_auto_sync()

    def _iter_supported_files(self, root_path):
        for current_root, _, files in os.walk(root_path):
            for name in files:
                if not _is_supported_knowledge_file(name):
                    continue
                yield os.path.abspath(os.path.join(current_root, name))

    def _path_signature(self, roots):
        rows = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for path in self._iter_supported_files(root):
                try:
                    stat = os.stat(path)
                    rows.append(f"{path}|{int(stat.st_mtime)}|{int(stat.st_size)}")
                except Exception:
                    continue
        payload = "\n".join(sorted(rows))
        return sha1(payload.encode("utf-8")).hexdigest()

    def _auto_sync_loop(self):
        logger.info(
            "KB auto-sync worker started (roots=%s, interval=%.1fs)",
            self._auto_sync_roots,
            self._auto_sync_interval_seconds,
        )
        while not self._auto_sync_stop_event.is_set():
            try:
                roots = list(self._auto_sync_roots)
                if roots:
                    signature = self._path_signature(roots)
                    if signature != self._auto_sync_last_signature:
                        indexed_total = 0
                        skipped_total = 0
                        removed_total = 0
                        for root in roots:
                            ok, message, indexed_files, skipped_files, removed_files = self.sync_directory(root)
                            if not ok:
                                self._auto_sync_last_error = str(message or "sync_failed")
                                logger.warning("KB auto-sync failed for %s: %s", root, message)
                                continue
                            indexed_total += int(indexed_files or 0)
                            skipped_total += int(skipped_files or 0)
                            removed_total += int(removed_files or 0)

                        self._auto_sync_last_signature = signature
                        self._auto_sync_last_changes = {
                            "indexed_files": indexed_total,
                            "skipped_files": skipped_total,
                            "removed_files": removed_total,
                        }
                        self._auto_sync_last_run_ts = time.time()
            except Exception as exc:
                self._auto_sync_last_error = str(exc)
                logger.warning("KB auto-sync loop failed: %s", exc)

            self._auto_sync_stop_event.wait(self._auto_sync_interval_seconds)

        logger.info("KB auto-sync worker stopped")

    # --- Watchdog support (preferred when available) ---------------------
    def _schedule_watch_sync(self, delay_seconds=0.25):
        # Debounce rapid filesystem events and run a single directory sync.
        with self._watchdog_debounce_lock:
            if self._watchdog_debounce_timer:
                try:
                    self._watchdog_debounce_timer.cancel()
                except Exception:
                    pass
            self._watchdog_debounce_timer = threading.Timer(delay_seconds, self._run_watch_sync)
            self._watchdog_debounce_timer.daemon = True
            self._watchdog_debounce_timer.start()

    def _run_watch_sync(self):
        try:
            roots = list(self._auto_sync_roots)
            if not roots:
                return
            signature = self._path_signature(roots)
            if signature == self._auto_sync_last_signature:
                return
            indexed_total = 0
            skipped_total = 0
            removed_total = 0
            for root in roots:
                try:
                    ok, message, indexed_files, skipped_files, removed_files = self.sync_directory(root)
                except Exception as exc:
                    logger.warning("KB watchdog sync failed for %s: %s", root, exc)
                    continue
                if not ok:
                    self._auto_sync_last_error = str(message or "sync_failed")
                    logger.warning("KB watchdog sync failed for %s: %s", root, message)
                    continue
                indexed_total += int(indexed_files or 0)
                skipped_total += int(skipped_files or 0)
                removed_total += int(removed_files or 0)

            self._auto_sync_last_signature = signature
            self._auto_sync_last_changes = {
                "indexed_files": indexed_total,
                "skipped_files": skipped_total,
                "removed_files": removed_total,
            }
            self._auto_sync_last_run_ts = time.time()
        except Exception as exc:
            self._auto_sync_last_error = str(exc)
            logger.warning("KB watchdog run sync failed: %s", exc)

    class _KBWatchHandler(FileSystemEventHandler):
        def __init__(self, service):
            super().__init__()
            self._service = service

        def on_any_event(self, event):
            try:
                self._service._schedule_watch_sync()
            except Exception:
                pass

    def _start_watchdog(self):
        if not _WATCHDOG_AVAILABLE:
            return False, "watchdog not available"
        with self._lock:
            if self._watchdog_observer is not None:
                return True, "KB watchdog already running"
            if not self._auto_sync_roots:
                return False, "KB auto-sync requires at least one existing directory"
            observer = Observer()
            handlers = []
            for root in self._auto_sync_roots:
                try:
                    handler = KnowledgeBaseService._KBWatchHandler(self)
                    observer.schedule(handler, path=root, recursive=True)
                    handlers.append(handler)
                except Exception as exc:
                    logger.warning("Failed scheduling watchdog for %s: %s", root, exc)
            observer.daemon = True
            observer.start()
            self._watchdog_observer = observer
            self._watchdog_handlers = handlers
        return True, "KB watchdog started"

    def _stop_watchdog(self):
        with self._lock:
            observer = self._watchdog_observer
            self._watchdog_observer = None
            self._watchdog_handlers = []
        if observer is not None:
            try:
                observer.stop()
                observer.join(timeout=1.0)
            except Exception:
                pass
        with self._watchdog_debounce_lock:
            if self._watchdog_debounce_timer:
                try:
                    self._watchdog_debounce_timer.cancel()
                except Exception:
                    pass
                self._watchdog_debounce_timer = None
        return True, "KB watchdog stopped"

    def set_auto_sync(self, enabled, roots=None, interval_seconds=None):
        self._ensure_initialized()
        with self._lock:
            self._auto_sync_enabled = bool(enabled)
            if roots is not None:
                cleaned = []
                for item in roots:
                    path = os.path.abspath(os.path.expanduser(str(item or "").strip()))
                    if path and os.path.isdir(path):
                        cleaned.append(path)
                self._auto_sync_roots = cleaned
            if interval_seconds is not None:
                self._auto_sync_interval_seconds = max(1.0, float(interval_seconds or 1.0))

        if self._auto_sync_enabled and self._auto_sync_roots:
            self.start_auto_sync()
        else:
            self.stop_auto_sync()

        return True, (
            f"KB auto-sync enabled={self._auto_sync_enabled}, "
            f"roots={len(self._auto_sync_roots)}, interval={self._auto_sync_interval_seconds:.1f}s"
        )

    def start_auto_sync(self):
        self._ensure_initialized()
        with self._lock:
            # Prefer watchdog-based event sync when available; otherwise fall back to polling thread.
            if _WATCHDOG_AVAILABLE:
                ok, msg = self._start_watchdog()
                if ok:
                    return True, msg
                # If watchdog failed to start, fall through to polling fallback.

            if self._auto_sync_thread is not None and self._auto_sync_thread.is_alive():
                return True, "KB auto-sync already running"
            if not self._auto_sync_roots:
                return False, "KB auto-sync requires at least one existing directory"
            self._auto_sync_stop_event.clear()
            self._auto_sync_thread = threading.Thread(
                target=self._auto_sync_loop,
                name="jarvis-kb-auto-sync",
                daemon=True,
            )
            self._auto_sync_thread.start()
        return True, "KB auto-sync started"

    def stop_auto_sync(self):
        with self._lock:
            thread = self._auto_sync_thread
            self._auto_sync_thread = None
        self._auto_sync_stop_event.set()
        # Stop watchdog if active
        if _WATCHDOG_AVAILABLE:
            try:
                self._stop_watchdog()
            except Exception:
                pass

        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        return True, "KB auto-sync stopped"

    def auto_sync_status(self):
        with self._lock:
            running = bool(self._auto_sync_thread is not None and self._auto_sync_thread.is_alive())
            roots = list(self._auto_sync_roots)
            interval = float(self._auto_sync_interval_seconds)
            enabled = bool(self._auto_sync_enabled)
            last_signature = self._auto_sync_last_signature
            last_run_ts = float(self._auto_sync_last_run_ts or 0.0)
            last_error = str(self._auto_sync_last_error or "")
            last_changes = dict(self._auto_sync_last_changes or {})
        return {
            "enabled": enabled,
            "running": running,
            "roots": roots,
            "interval_seconds": interval,
            "last_signature": last_signature,
            "last_run_ts": last_run_ts,
            "last_error": last_error,
            "last_changes": last_changes,
        }

    def _load_json_file(self, path, default_value):
        if not os.path.isfile(path):
            return default_value
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload
        except Exception as exc:
            logger.error("Failed loading JSON file %s: %s", path, exc)
            return default_value

    def _load_from_disk(self):
        loaded_meta = self._load_json_file(KB_META_FILE, [])
        self._meta = loaded_meta if isinstance(loaded_meta, list) else []

        loaded_sources = self._load_json_file(KB_SOURCE_STATE_FILE, {})
        self._source_state = loaded_sources if isinstance(loaded_sources, dict) else {}

        if os.path.isfile(self._vectors_file):
            try:
                self._vectors = np.load(self._vectors_file).astype(np.float32)
            except Exception as exc:
                logger.error("Failed loading vectors file: %s", exc)
                self._vectors = np.zeros((0, self._embedder.dimension), dtype=np.float32)

        if self._vectors.size == 0 and faiss is not None and os.path.isfile(KB_FAISS_INDEX_FILE):
            try:
                self._faiss_index = faiss.read_index(KB_FAISS_INDEX_FILE)
                logger.info("Loaded FAISS KB index with %s vectors", int(self._faiss_index.ntotal))
            except Exception as exc:
                logger.error("Failed loading FAISS index: %s", exc)
                self._faiss_index = None
        else:
            self._rebuild_faiss_index()

    def _save_to_disk(self):
        with open(KB_META_FILE, "w", encoding="utf-8") as handle:
            json.dump(self._meta, handle, ensure_ascii=True, indent=2)

        with open(KB_SOURCE_STATE_FILE, "w", encoding="utf-8") as handle:
            json.dump(self._source_state, handle, ensure_ascii=True, indent=2)

        np.save(self._vectors_file, self._vectors)
        if self._faiss_index is not None and faiss is not None:
            faiss.write_index(self._faiss_index, KB_FAISS_INDEX_FILE)

    def _rebuild_faiss_index(self):
        if faiss is None:
            self._faiss_index = None
            return
        if self._vectors.size == 0:
            self._faiss_index = None
            return
        index = faiss.IndexFlatIP(self._vectors.shape[1])
        index.add(self._vectors)
        self._faiss_index = index

    def _next_id(self):
        if not self._meta:
            return 0
        return max(int(row.get("id", 0)) for row in self._meta) + 1

    def _split_text(self, text):
        body = (text or "").strip()
        if not body:
            return []
        size = max(120, int(KB_CHUNK_SIZE))
        overlap = max(0, min(size - 1, int(KB_CHUNK_OVERLAP)))
        chunks = []
        start = 0
        length = len(body)
        while start < length:
            end = min(length, start + size)
            chunk = body[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= length:
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _read_document(self, path):
        lower = path.lower()
        if lower.endswith(".pdf"):
            text = _safe_read_pdf(path)
            if text:
                return text
        return _safe_read_text_file(path)

    def _hash_text(self, text):
        return sha256((text or "").encode("utf-8")).hexdigest()

    def _hash_file_content(self, text):
        return sha256((text or "").encode("utf-8")).hexdigest()

    def _remove_source_chunks_locked(self, source_path):
        if not self._meta:
            return 0

        keep_indices = [i for i, row in enumerate(self._meta) if row.get("source") != source_path]
        removed = len(self._meta) - len(keep_indices)
        if removed <= 0:
            return 0

        self._meta = [self._meta[i] for i in keep_indices]
        if self._vectors.size > 0:
            self._vectors = self._vectors[keep_indices, :]
        self._source_state.pop(source_path, None)
        self._rebuild_faiss_index()
        return removed

    def _append_records_locked(self, source_path, chunks, vectors, content_hash, mtime):
        base_id = self._next_id()
        for chunk_index, chunk_text in enumerate(chunks):
            self._meta.append(
                {
                    "id": base_id + chunk_index,
                    "source": source_path,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "chunk_hash": self._hash_text(chunk_text),
                    "content_hash": content_hash,
                    "indexed_at": time.time(),
                }
            )

        if self._vectors.size == 0:
            self._vectors = vectors.astype(np.float32)
        else:
            self._vectors = np.vstack([self._vectors, vectors]).astype(np.float32)
        self._rebuild_faiss_index()

        self._source_state[source_path] = {
            "content_hash": content_hash,
            "mtime": _safe_float(mtime),
            "chunk_count": len(chunks),
            "indexed_at": time.time(),
        }

    def add_document(self, path):
        self._ensure_initialized()
        if not self._enabled:
            return False, "Knowledge base is disabled.", 0

        resolved = os.path.abspath(os.path.expanduser((path or "").strip().strip('"').strip("'")))
        if not os.path.isfile(resolved):
            return False, f"File not found: {resolved}", 0

        try:
            text = self._read_document(resolved)
        except Exception as exc:
            return False, f"Failed reading file: {exc}", 0

        chunks = self._split_text(text)
        if not chunks:
            return False, "No readable text found in file.", 0

        content_hash = self._hash_file_content(text)
        mtime = _safe_float(os.path.getmtime(resolved))

        with self._lock:
            source_info = self._source_state.get(resolved, {})
            if source_info.get("content_hash") == content_hash:
                return True, f"Document unchanged; skipped re-index: {resolved}", 0

            self._remove_source_chunks_locked(resolved)
            vectors = self._embedder.embed(chunks)
            self._append_records_locked(
                source_path=resolved,
                chunks=chunks,
                vectors=vectors,
                content_hash=content_hash,
                mtime=mtime,
            )
            self._save_to_disk()

        return True, f"Indexed {len(chunks)} chunk(s) from {resolved}", len(chunks)

    def index_directory(self, root_path):
        self._ensure_initialized()
        root = os.path.abspath(os.path.expanduser((root_path or "").strip().strip('"').strip("'")))
        if not os.path.isdir(root):
            return False, f"Directory does not exist: {root}", 0, 0

        indexed_files = 0
        indexed_chunks = 0
        for current_root, _, files in os.walk(root):
            for name in files:
                if not _is_supported_knowledge_file(name):
                    continue
                ok, message, count = self.add_document(os.path.join(current_root, name))
                if ok and count > 0:
                    indexed_files += 1
                    indexed_chunks += count
                elif not ok:
                    logger.warning("KB index file failed: %s (%s)", os.path.join(current_root, name), message)
        return True, f"Indexed {indexed_files} file(s), {indexed_chunks} chunk(s).", indexed_files, indexed_chunks

    def sync_directory(self, root_path):
        self._ensure_initialized()
        root = os.path.abspath(os.path.expanduser((root_path or "").strip().strip('"').strip("'")))
        if not os.path.isdir(root):
            return False, f"Directory does not exist: {root}", 0, 0, 0

        indexed_files = 0
        skipped_files = 0
        indexed_chunks = 0
        discovered_files = set()

        for current_root, _, files in os.walk(root):
            for name in files:
                if not _is_supported_knowledge_file(name):
                    continue
                path = os.path.abspath(os.path.join(current_root, name))
                discovered_files.add(path)
                ok, _message, count = self.add_document(path)
                if ok and count > 0:
                    indexed_files += 1
                    indexed_chunks += count
                elif ok and count == 0:
                    skipped_files += 1

        removed_files = 0
        with self._lock:
            known_sources = list(self._source_state.keys())
            for source in known_sources:
                if source.startswith(root + os.sep) and source not in discovered_files:
                    removed = self._remove_source_chunks_locked(source)
                    if removed > 0:
                        removed_files += 1
            self._save_to_disk()

        return (
            True,
            (
                f"Sync complete: indexed_files={indexed_files}, skipped_files={skipped_files}, "
                f"removed_files={removed_files}, indexed_chunks={indexed_chunks}"
            ),
            indexed_files,
            skipped_files,
            removed_files,
        )

    def _search_vectors(self, query_vector, top_n):
        if self._faiss_index is not None and faiss is not None:
            scores, indices = self._faiss_index.search(query_vector, top_n)
            return list(zip(indices[0].tolist(), scores[0].tolist()))

        if self._vectors.size == 0:
            return []
        similarities = self._vectors @ query_vector[0]
        top_indices = np.argsort(-similarities)[:top_n]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def _lexical_score(self, query, text):
        query_tokens = set(_TOKEN_RE.findall((query or "").lower()))
        text_tokens = set(_TOKEN_RE.findall((text or "").lower()))
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(text_tokens)) / float(len(query_tokens))
        phrase_bonus = 0.2 if (query or "").lower() in (text or "").lower() else 0.0
        return min(1.0, overlap + phrase_bonus)

    def _normalize_vector_score(self, value):
        # Similarity scores are roughly in [-1, 1] for cosine-like vectors.
        return max(0.0, min(1.0, (float(value) + 1.0) / 2.0))

    def search(self, query, top_k=KB_TOP_K):
        self._ensure_initialized()
        if not self._enabled:
            return []
        if not self._meta or self._vectors.size == 0:
            return []

        top_k = max(1, int(top_k))
        candidate_n = max(top_k, top_k * max(1, int(KB_RERANK_CANDIDATE_MULTIPLIER)))
        query_vector = self._embedder.embed([query])
        raw_pairs = self._search_vectors(query_vector, top_n=candidate_n)

        ranked = []
        for idx, vector_score in raw_pairs:
            if idx < 0 or idx >= len(self._meta):
                continue
            row = self._meta[idx]
            lexical_score = self._lexical_score(query, row.get("text", ""))
            normalized_vector = self._normalize_vector_score(vector_score)
            combined = (
                float(KB_EMBEDDING_RERANK_WEIGHT) * normalized_vector
                + float(KB_LEXICAL_RERANK_WEIGHT) * lexical_score
            )
            ranked.append(
                {
                    "id": row["id"],
                    "score": float(combined),
                    "vector_score": float(vector_score),
                    "lexical_score": float(lexical_score),
                    "source": row["source"],
                    "chunk_index": row["chunk_index"],
                    "text": row["text"],
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_k]

    def _sanitize_context_text(self, text):
        sanitized = (text or "").strip()
        lowered = sanitized.lower()
        for pattern in KB_BLOCKED_CONTEXT_PATTERNS:
            if pattern in lowered:
                sanitized = re.sub(
                    re.escape(pattern),
                    "[redacted-instruction]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
                lowered = sanitized.lower()

        lines = []
        for line in sanitized.splitlines():
            cleaned = _SUSPICIOUS_PREFIX_RE.sub("", line).strip()
            if cleaned:
                lines.append(cleaned)
        if not lines:
            return ""
        return " ".join(lines)

    def retrieve_for_prompt(self, query, top_k=KB_TOP_K, max_chars=KB_MAX_CONTEXT_CHARS):
        if not self.is_retrieval_enabled():
            return {"context": "", "sources": [], "results": []}

        results = self.search(query, top_k=top_k)
        if not results:
            return {"context": "", "sources": [], "results": []}

        query_tokens = set(_TOKEN_RE.findall((query or "").lower()))
        lines = []
        sources = []
        chars = 0
        for item in results:
            score = float(item.get("score", 0.0))
            lexical_score = float(item.get("lexical_score", 0.0))
            if score < float(KB_MIN_PROMPT_SCORE):
                continue
            if (
                len(query_tokens) >= 2
                and lexical_score <= 0.0
                and score < float(KB_MIN_SEMANTIC_ONLY_SCORE)
            ):
                continue

            snippet = self._sanitize_context_text(item["text"])
            if not snippet:
                continue

            rank = len(sources) + 1
            line = f"[{rank}] source={item['source']} chunk={item['chunk_index']} :: {snippet}"
            if chars + len(line) > max_chars:
                break
            lines.append(line)
            sources.append(
                {
                    "rank": rank,
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "score": score,
                }
            )
            chars += len(line)

        return {
            "context": "\n".join(lines),
            "sources": sources,
            "results": results,
        }

    def build_context(self, query, top_k=KB_TOP_K, max_chars=KB_MAX_CONTEXT_CHARS):
        package = self.retrieve_for_prompt(query, top_k=top_k, max_chars=max_chars)
        return package["context"]

    def clear(self):
        self._ensure_initialized()
        with self._lock:
            self._meta = []
            self._vectors = np.zeros((0, self._embedder.dimension), dtype=np.float32)
            self._faiss_index = None
            self._source_state = {}
            self._save_to_disk()
        return True, "Knowledge base cleared."

    def set_retrieval_enabled(self, enabled):
        self._ensure_initialized()
        with self._lock:
            self._retrieval_enabled = bool(enabled)
        return True, f"Knowledge base retrieval set to: {bool(enabled)}"

    def is_retrieval_enabled(self):
        self._ensure_initialized()
        with self._lock:
            return self._retrieval_enabled

    def status(self):
        self._ensure_initialized()
        with self._lock:
            chunk_count = len(self._meta)
            file_count = len(self._source_state)
            retrieval = self._retrieval_enabled
            enabled = self._enabled
            vector_backend = "faiss" if self._faiss_index is not None else "numpy"
            embedding_backend = self._embedder.mode
            dim = self._embedder.dimension
        return {
            "enabled": enabled,
            "retrieval_enabled": retrieval,
            "vector_backend": vector_backend,
            "embedding_backend": embedding_backend,
            "semantic_backend_ready": embedding_backend != "hash",
            "embedding_dim": dim,
            "chunk_count": chunk_count,
            "file_count": file_count,
            "storage_dir": os.path.abspath(KB_STORAGE_DIR),
            "source_state_file": os.path.abspath(KB_SOURCE_STATE_FILE),
            "auto_sync": self.auto_sync_status(),
        }

    def quality_report(self, probe_count=20, top_k=3):
        self._ensure_initialized()
        status = self.status()
        if status["chunk_count"] == 0:
            return {
                "ok": False,
                "reason": "empty_knowledge_base",
                "probes": 0,
                "accuracy_at_k": 0.0,
                "status": status,
            }

        rows = list(self._meta[: max(1, int(probe_count))])
        probes = []
        hits = 0
        for row in rows:
            tokens = [token for token in _TOKEN_RE.findall((row.get("text") or "").lower()) if len(token) >= 4]
            query = " ".join(tokens[:3]) if tokens else (row.get("text") or "")[:40]
            if not query:
                continue
            results = self.search(query, top_k=max(1, int(top_k)))
            expected_source = row.get("source")
            matched = any(item.get("source") == expected_source for item in results)
            if matched:
                hits += 1
            probes.append(
                {
                    "query": query,
                    "expected_source": expected_source,
                    "matched": matched,
                }
            )

        total = len(probes)
        accuracy = (hits / total) if total else 0.0
        return {
            "ok": True,
            "probes": total,
            "hits": hits,
            "accuracy_at_k": accuracy,
            "status": status,
            "semantic_backend_ready": status["semantic_backend_ready"],
            "probe_preview": probes[:5],
        }


knowledge_base_service = KnowledgeBaseService()
