import os
import sqlite3
import json
import numpy as np
import uuid

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
                results = self.collection.query(query_texts=[task], n_results=n_results * 2)  # Fetch extra to filter
                if results and results.get('documents') and results['documents'][0]:
                    docs = results['documents'][0]
                    metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                    
                    # Prioritize solution memories over mistake logs
                    solutions = []
                    mistakes = []
                    for i, doc in enumerate(docs):
                        meta = metadatas[i] if i < len(metadatas) else {}
                        if meta.get('type') == 'mistake_fix':
                            mistakes.append(doc)
                        else:
                            solutions.append(doc)
                    
                    # Use solutions first, add at most 1 mistake pattern for context
                    filtered = solutions[:n_results]
                    if len(filtered) < n_results and mistakes:
                        filtered.append(mistakes[0])
                    
                    if not filtered:
                        filtered = docs[:n_results]
                    
                    memories = "\n---\n".join(filtered)
                    # Limit memory injection to prevent Context Window OOM
                    if len(memories) > 3000:
                        cutoff = memories.rfind('\n\n', 0, 3000)
                        cutoff = cutoff if cutoff != -1 else 3000
                        memories = memories[:cutoff] + "\n\n... [TRUNCATED]"
                    return f"\n\nRelevant past experience:\n{memories}\n"
            except Exception as e:
                print(f"ChromaDB query failed: {str(e)}. Falling back to SQLite recall.")

        # SQLite Query Fallback
        with sqlite3.connect(self.sqlite_path, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, task, doc, metadata, embedding FROM memories")
            rows = cursor.fetchall()

        if not rows:
            return ""

        def _prioritize_and_format(scored_items, limit):
            solutions = []
            mistakes = []
            for score, doc, meta_str in scored_items:
                meta = {}
                if meta_str:
                    try:
                        meta = json.loads(meta_str)
                    except:
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

        # Vector search if embed_fn is available and we have embeddings
        if embed_fn and rows[0][4]:
            try:
                query_vector = np.array(embed_fn(task))
                norm_q = np.linalg.norm(query_vector)  # Compute ONCE outside the loop
                if norm_q == 0:
                    norm_q = 1e-10  # Avoid division by zero
                scores = []
                for mem_id, t_task, doc, meta_str, emb_blob in rows:
                    if emb_blob:
                        emb = np.frombuffer(emb_blob, dtype=np.float32)  # Binary decode (fast)
                        norm_e = np.linalg.norm(emb)
                        if norm_e > 0:
                            similarity = np.dot(query_vector, emb) / (norm_q * norm_e)
                        else:
                            similarity = 0
                        scores.append((similarity, doc, meta_str))
                
                # Sort by similarity descending
                scores.sort(key=lambda x: x[0], reverse=True)
                scored_items = [item for item in scores if item[0] > 0.4] # threshold
                if scored_items:
                    return _prioritize_and_format(scored_items, n_results)
            except Exception as e:
                print(f"SQLite vector similarity search failed: {str(e)}")

        # Text keyword search fallback with stopword filtering to avoid false positives
        STOPWORDS = {
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
            # Common task-agnostic coding verbs to ignore
            "write", "code", "program", "script", "create", "make", "generate", "give", "please", "solve", "run",
            "plot", "plots", "plotting", "3d", "2d", "verify", "verification", "mathematically", "mathematical", 
            "equation", "equations", "function", "functions", "surface", "constant", "constants", "value", "values", 
            "parameter", "parameters", "result", "results", "output", "show", "showing", "find", "prove", "theorem", 
            "calculate", "derivation", "derive", "analytical", "numerical", "scenario"
        }
        
        query_words = set(w.strip(",.!?") for w in task.lower().split() if w not in STOPWORDS and len(w) > 1)
        if not query_words:
            return ""

        keyword_scores = []
        for mem_id, t_task, doc, meta_str, _ in rows:
            task_words = set(w.strip(",.!?") for w in t_task.lower().split() if w not in STOPWORDS)
            # Find matching keywords
            matches = len(query_words.intersection(task_words))
            # Require at least 2 matching significant words, or 100% of the query terms if query is very short
            if matches >= 2 or (len(query_words) == 1 and matches == 1):
                keyword_scores.append((matches, doc, meta_str))
                
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
            import re
            return set(re.findall(r'\b\d+(?:\.\d+)?\b', t))

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
            if overlap > 0.8:  # Higher threshold for content words to prevent false duplicate matching
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

        # 2. Key procedure (first 500 chars of code as a summary, not the full thing)
        code_summary = successful_code[:500].strip()
        if len(successful_code) > 500:
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

