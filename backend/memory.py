import os
import sqlite3
import json
import math
import time as _time
import numpy as np
import uuid
import re

# ─────────────────────────────────────────────────────────────────────────
# Retrieval-quality knobs (Phase 2.1)
# These are the empirical cutoffs that separate "genuinely related" hits
# from "vaguely on-topic" noise for MiniLM-class 384-d embeddings.
#
# MIN_VECTOR_SCORE          — hard floor for a hit to count as relevant.
# TOP_HIT_MARGIN            — drop any hit more than this cosine below the
#                              best hit (prevents blending 0.42s with 0.91s).
# RECENCY_WEIGHT            — how much a recent memory outranks an old one
#                              of equal cosine similarity. 0.15 == "modest".
# RECENCY_HALF_LIFE_DAYS    — half-life for the recency bonus.
# KEYWORD_MIN_MATCHES       — content-word overlap required for the
#                              keyword fallback to fire when vector search
#                              produces no hits above MIN_VECTOR_SCORE.
# ─────────────────────────────────────────────────────────────────────────
MIN_VECTOR_SCORE = 0.65
TOP_HIT_MARGIN = 0.15
RECENCY_WEIGHT = 0.15
RECENCY_HALF_LIFE_DAYS = 30.0
KEYWORD_MIN_MATCHES = 3

class Memory:
    def __init__(self, db_path="./forge_memory_db"):
        self.db_path = db_path
        self.chroma_client = None
        self.collection = None
        self.use_chroma = False

        # Attempt to initialize ChromaDB
        try:
            import chromadb
            os.makedirs(db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_or_create_collection(name="knowledge")
            self.use_chroma = True
            print("Memory Engine: Successfully initialized ChromaDB persistent vector database.")
        except Exception as e:
            print(f"Memory Engine: ChromaDB not available ({str(e)}). Falling back to SQLite memory store.")
            
        # Initialize SQLite database (either as primary fallback or metadata companion)
        self.sqlite_path = os.path.join(db_path, "local_memory.db")
        os.makedirs(db_path, exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize local SQLite database for structured data and embedding storage."""
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            
            # Table for storing experiences (tasks, solutions, mistakes)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    task TEXT,
                    doc TEXT,
                    metadata TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def count(self):
        """Returns the number of stored memories."""
        if self.use_chroma:
            try:
                return self.collection.count()
            except Exception:
                pass
        
        # SQLite count
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM memories")
            count = cursor.fetchone()[0]
        return count

    def recall(self, task, n_results=2, embed_fn=None):
        """
        Search memory for past experiences related to the current task.
        Prioritizes successful solution patterns over mistake/error logs to
        prevent error contamination when the same question is asked again.
        """
        if self.count() == 0:
            return ""

        # Try ChromaDB query first
        if self.use_chroma:
            try:
                results = self.collection.query(
                    query_texts=[task],
                    n_results=n_results * 2,  # Fetch extra to filter
                    include=["documents", "metadatas", "distances"]
                )
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                    distances = results.get('distances', [[]])[0] if results.get('distances') else []
                    
                    # ChromaDB returns L2 distances by default. Convert to
                    # approximate cosine similarity: cos ≈ 1 - (d² / 2) for
                    # unit-normed embeddings. Filter out noise below threshold.
                    filtered_docs = []
                    for i, doc in enumerate(docs):
                        if i < len(distances):
                            l2_dist = distances[i]
                            approx_cosine = 1.0 - (l2_dist ** 2) / 2.0
                            if approx_cosine < MIN_VECTOR_SCORE:
                                continue  # Below similarity threshold — noise
                        meta = metadatas[i] if i < len(metadatas) else {}
                        filtered_docs.append((doc, meta))
                    
                    if not filtered_docs:
                        # No hits above threshold — fall through to keyword search
                        pass
                    else:
                        # Prioritize solution memories over mistake logs
                        solutions = []
                        mistakes = []
                        for doc, meta in filtered_docs:
                            if meta.get('type') == 'mistake_fix':
                                mistakes.append(doc)
                            else:
                                solutions.append(doc)
                        
                        # Use solutions first, add at most 1 mistake pattern for context
                        filtered = solutions[:n_results]
                        if len(filtered) < n_results and mistakes:
                            filtered.append(mistakes[0])
                        
                        if not filtered:
                            filtered = [d for d, _ in filtered_docs[:n_results]]
                        
                        memories = "\n---\n".join(filtered)
                        # Limit memory injection to prevent Context Window OOM
                        if len(memories) > 3000:
                            cutoff = memories.rfind('\n\n', 0, 3000)
                            cutoff = cutoff if cutoff != -1 else 3000
                            memories = memories[:cutoff] + "\n\n... [TRUNCATED]"
                        return f"\n\nRelevant past experience:\n{memories}\n"
            except Exception as e:
                print(f"ChromaDB query failed: {str(e)}. Falling back to SQLite recall.")

        # SQLite Query Fallback — now selects `created_at` so we can compute
        # a recency bonus (Phase 2.1).
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, task, doc, metadata, embedding, created_at FROM memories"
            )
            rows = cursor.fetchall()

        if not rows:
            return ""

        def _prioritize_and_format(scored_items, limit):
            """
            scored_items is a list of (score, doc, meta_str). Higher-score items
            already come first. This function:
              (a) drops any hit more than TOP_HIT_MARGIN below the best score;
              (b) preserves solution > mistake ordering;
              (c) trims to `limit` items and truncates the joined text.
            """
            if not scored_items:
                return ""

            best_score = scored_items[0][0]
            filtered_by_margin = [
                item for item in scored_items
                if (best_score - item[0]) <= TOP_HIT_MARGIN
            ]

            solutions = []
            mistakes = []
            for score, doc, meta_str in filtered_by_margin:
                meta = {}
                if meta_str:
                    try:
                        meta = json.loads(meta_str)
                    except Exception:
                        pass
                if meta.get('type') == 'mistake_fix':
                    mistakes.append(doc)
                else:
                    solutions.append(doc)

            filtered = solutions[:limit]
            if len(filtered) < limit and mistakes:
                filtered.append(mistakes[0])

            if not filtered:
                return ""

            memories = "\n---\n".join(filtered)
            if len(memories) > 4000:
                cutoff = memories.rfind('\n\n', 0, 4000)
                cutoff = cutoff if cutoff != -1 else 4000
                memories = memories[:cutoff] + "\n\n... [TRUNCATED]"
            return f"\n\nRelevant past experience:\n{memories}\n"

        def _recency_bonus(created_at_str):
            """
            Convert an ISO-8601 `created_at` timestamp into a recency multiplier
            in the interval (0, 1]. A memory saved *right now* returns 1.0;
            RECENCY_HALF_LIFE_DAYS ago returns 0.5; a year ago ≈ 0.001.
            Returns 0.5 (neutral) if the timestamp is unparseable.
            """
            if not created_at_str:
                return 0.5
            try:
                # SQLite CURRENT_TIMESTAMP format: "YYYY-MM-DD HH:MM:SS"
                import datetime as _dt
                ts = _dt.datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S")
                age_days = (_dt.datetime.utcnow() - ts).total_seconds() / 86400.0
                age_days = max(0.0, age_days)
                return math.exp(-age_days * math.log(2) / RECENCY_HALF_LIFE_DAYS)
            except Exception:
                return 0.5

        # ── Vector search (primary) ───────────────────────────────────────
        if embed_fn and rows[0][4]:
            try:
                query_vector = np.array(embed_fn(task))
                norm_q = np.linalg.norm(query_vector)
                if norm_q == 0:
                    norm_q = 1e-10
                scores = []
                for mem_id, t_task, doc, meta_str, emb_blob, created_at in rows:
                    if not emb_blob:
                        continue
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    norm_e = np.linalg.norm(emb)
                    if norm_e <= 0:
                        continue
                    cosine = float(np.dot(query_vector, emb) / (norm_q * norm_e))
                    if cosine < MIN_VECTOR_SCORE:
                        # Hard-drop noise below the "genuinely related" floor.
                        continue
                    # Blend cosine similarity with a recency bonus so that a
                    # slightly-less-similar but newer memory can beat an old
                    # one on ties. Weight is deliberately small (0.15) so
                    # semantic match still dominates.
                    final = (1.0 - RECENCY_WEIGHT) * cosine + \
                            RECENCY_WEIGHT * _recency_bonus(created_at)
                    scores.append((final, doc, meta_str))

                if scores:
                    scores.sort(key=lambda x: x[0], reverse=True)
                    return _prioritize_and_format(scores, n_results)
                # Fall through to keyword search ONLY when vector search
                # produced zero above-threshold hits.
            except Exception as e:
                print(f"SQLite vector similarity search failed: {str(e)}")

        # ── Keyword-overlap fallback ─────────────────────────────────────
        # Trimmed STOPWORDS: `plot`, `equation`, `theorem`, `derive`, `3d`,
        # etc. are actually strong topical signals and should NOT be silenced.
        # To compensate for the wider vocabulary, we require KEYWORD_MIN_MATCHES
        # (currently 3) content-word matches — up from the old threshold of 2.
        STOPWORDS = {
            # Grammar-only stopwords
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at",
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
            "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd",
            "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
            "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
            "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
            "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
            "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
            "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
            "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
            # Purely task-agnostic verbs — these carry no topical signal
            "write", "code", "program", "script", "create", "make", "generate", "give", "please", "solve", "run",
            "show", "showing", "output", "result", "results", "value", "values",
            # NOTE: `plot`, `equation`, `theorem`, `derive`, `3d`, `numerical`,
            # `analytical`, `verify`, `mathematical`, `function`, `surface`,
            # `parameter`, `constant`, `find`, `calculate`, `scenario` are
            # INTENTIONALLY NOT in stopwords — they are topical anchors.
        }

        query_words = set(
            w.strip(",.!?") for w in task.lower().split()
            if w not in STOPWORDS and len(w) > 1
        )
        if not query_words:
            return ""

        keyword_scores = []
        for row in rows:
            mem_id, t_task, doc, meta_str, _emb, _created_at = row
            task_words = set(
                w.strip(",.!?") for w in t_task.lower().split()
                if w not in STOPWORDS and len(w) > 1
            )
            matches = len(query_words.intersection(task_words))
            # Require KEYWORD_MIN_MATCHES (3) overlapping content words. For
            # short queries (1-2 content words) we still require full overlap
            # to prevent single-word coincidences from surfacing unrelated
            # memories as "relevant past experience".
            if matches >= KEYWORD_MIN_MATCHES or (
                len(query_words) <= 2 and matches == len(query_words)
            ):
                keyword_scores.append((float(matches), doc, meta_str))

        if keyword_scores:
            keyword_scores.sort(key=lambda x: x[0], reverse=True)
            return _prioritize_and_format(keyword_scores, n_results)

        return ""

    def _is_duplicate(self, task):
        """Check if a very similar task already exists in memory."""
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT task FROM memories")
            rows = cursor.fetchall()

        # Stopwords list to filter out generic noise words
        STOPWORDS = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", 
            "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", 
            "can", "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from", 
            "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", 
            "i", "if", "in", "into", "is", "it", "its", "me", "more", "most", "my", "myself", "no", "nor", 
            "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "out", "over", "own", 
            "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", 
            "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", 
            "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", 
            "whom", "why", "with", "you", "your", "yours", "yourself", "yourselves"
        }

        # Filter out punctuation and stopwords to compare only content words
        def _get_content_words(t):
            words = [w.strip(",.!?()\"';:") for w in t.lower().split()]
            return set(w for w in words if w and w not in STOPWORDS)
            
        def _get_numbers(t):
            return set(re.findall(r'-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b', t))

        task_words = _get_content_words(task)
        task_nums = _get_numbers(task)
        if not task_words:
            return False

        for (existing_task,) in rows:
            existing_words = _get_content_words(existing_task)
            existing_nums = _get_numbers(existing_task)
            
            if not existing_words:
                continue
            
            # If the numeric parameters differ, it's a completely unique physics/math problem
            if task_nums != existing_nums:
                continue
                
            overlap = len(task_words & existing_words) / max(len(task_words), len(existing_words))
            if overlap > 0.95:  # Higher threshold for content words to prevent false duplicate matching
                return True
        return False

    def save(self, task, successful_code, metadata=None, embed_fn=None):
        """Save a compact knowledge summary (NOT the full code) to long-term memory."""
        # Skip if we already have a very similar task stored
        if self._is_duplicate(task):
            return None

        mem_id = f"mem_{uuid.uuid4().hex}"

        # Extract compact knowledge instead of dumping raw code
        # 1. Libraries used
        imports = [line.strip() for line in successful_code.split("\n")
                   if line.strip().startswith(("import ", "from "))]
        libs = ", ".join(imports[:5]) if imports else "standard library"

        # 2. Extract python code block if present, otherwise save the mathematical derivation
        code_summary = ""
        if "```python" in successful_code:
            try:
                start = successful_code.find("```python") + 9
                end = successful_code.find("```", start)
                if end != -1:
                    code_summary = "VERIFIED SCRIPT:\n" + successful_code[start:end].strip()
            except Exception:
                pass
                
        if not code_summary:
            code_summary = successful_code[:2500].strip()
            if len(successful_code) > 2500:
                code_summary += "\n... [truncated]"

        doc = (
            f"Task: {task}\n"
            f"Libraries: {libs}\n"
            f"Procedure Summary:\n{code_summary}"
        )

        meta = metadata if metadata else {"task": task, "type": "solution"}

        # Save to Chroma
        if self.use_chroma:
            try:
                self.collection.add(documents=[doc], metadatas=[meta], ids=[mem_id])
            except Exception as e:
                print(f"Chroma save failed: {str(e)}")

        # Save to SQLite (embeddings stored as binary blobs for fast retrieval)
        emb_blob = None
        if embed_fn:
            try:
                emb = embed_fn(task)
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Embedding generation failed: {str(e)}")

        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, task, doc, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (mem_id, task, doc, json.dumps(meta), emb_blob)
            )
            conn.commit()

        return mem_id

    def save_mistake(self, task, wrong_code, error_log, fixed_code, embed_fn=None):
        """Save a compact mistake-fix pattern (NOT full code) to prevent regression."""
        mem_id = f"mistake_{uuid.uuid4().hex}"

        # Extract only the error pattern and the fix insight, not full code dumps
        # 1. Error essence: first 300 chars of the error (usually the traceback line)
        error_essence = error_log.strip()[:300]

        # 2. What changed: diff-like summary (just the key lines that differ)
        wrong_lines = set(wrong_code.strip().split("\n"))
        fixed_lines = set(fixed_code.strip().split("\n"))
        removed = list(wrong_lines - fixed_lines)[:5]
        added = list(fixed_lines - wrong_lines)[:5]

        doc = (
            f"Task: {task}\n"
            f"Error: {error_essence}\n"
            f"Root Cause (removed lines): {'; '.join(l.strip() for l in removed) if removed else 'structural change'}\n"
            f"Fix Pattern (added lines): {'; '.join(l.strip() for l in added) if added else 'structural change'}"
        )

        meta = {"task": task, "type": "mistake_fix"}

        # Save to Chroma
        if self.use_chroma:
            try:
                self.collection.add(documents=[doc], metadatas=[meta], ids=[mem_id])
            except Exception as e:
                print(f"Chroma save mistake failed: {str(e)}")

        # Save to SQLite (embeddings stored as binary blobs for fast retrieval)
        emb_blob = None
        if embed_fn:
            try:
                emb = embed_fn(task)
                emb_blob = np.array(emb, dtype=np.float32).tobytes()
            except Exception as e:
                print(f"Embedding generation failed: {str(e)}")

        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO memories (id, task, doc, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
                (mem_id, task, doc, json.dumps(meta), emb_blob)
            )
            conn.commit()

        return mem_id

