import os
import ast
import re

try:
    import psutil
except ImportError:
    psutil = None

class RepoMapGenerator:
    """
    Generates a highly compressed Abstract Syntax Tree (AST) map of a repository.
    Supports Python, JavaScript, TypeScript, JSX, and TSX files.
    Includes dynamic resource-aware context clamping to prevent VRAM/RAM overflow.
    """
    def __init__(self, root_dir, ignore_dirs=None):
        self.root_dir = os.path.abspath(root_dir)
        self.ignore_dirs = set(ignore_dirs or ['.git', '__pycache__', 'venv', 'env', 'node_modules', 'dist', 'build', '.mypy_cache', '.pytest_cache'])

    def _parse_file(self, file_path):
        """Route to appropriate parser based on file extension."""
        if file_path.endswith('.py'):
            return self._parse_python_file(file_path)
        elif file_path.endswith(('.js', '.jsx', '.ts', '.tsx')):
            return self._parse_js_ts_file(file_path)
        return []

    def _parse_python_file(self, file_path):
        """Parse a single Python file and return its structural signature."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except Exception:
            return []

        lines = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                for sub_node in node.body:
                    if isinstance(sub_node, ast.FunctionDef) or isinstance(sub_node, ast.AsyncFunctionDef):
                        args = [arg.arg for arg in sub_node.args.args]
                        args_str = ", ".join(args)
                        lines.append(f"    def {sub_node.name}({args_str})")
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                args = [arg.arg for arg in node.args.args]
                args_str = ", ".join(args)
                lines.append(f"def {node.name}({args_str})")
                
        return lines

    def _parse_js_ts_file(self, file_path):
        """Extract classes, functions, and key arrow function exports from JS/TS source files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return []

        lines = []
        
        # 1. Match class definitions
        class_matches = re.finditer(r'(?:export\s+)?class\s+(\w+)', content)
        for m in class_matches:
            lines.append(f"class {m.group(1)}:")
            
        # 2. Match standard function declarations
        func_matches = re.finditer(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)', content)
        for m in func_matches:
            args = [a.strip().split(':')[0].strip() for a in m.group(2).split(',') if a.strip()]
            args_str = ", ".join(args)
            lines.append(f"function {m.group(1)}({args_str})")
            
        # 3. Match arrow functions / React components
        arrow_matches = re.finditer(r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>', content)
        for m in arrow_matches:
            args = [a.strip().split(':')[0].strip() for a in m.group(2).split(',') if a.strip()]
            args_str = ", ".join(args)
            lines.append(f"const {m.group(1)} = ({args_str}) => ...")

        return lines

    def generate_map(self, prompt=None, max_chars=24000):
        """Walks the repository, scores files based on relevance to the prompt,
        and generates a prioritized AST map up to dynamically clamped max_chars.
        """
        # Dynamic Resource-Aware Context Clamping
        try:
            import torch
            if torch.cuda.is_available():
                # Fetch free GPU memory on device 0
                free_vram_bytes, _ = torch.cuda.mem_get_info(0)
                free_vram_gb = free_vram_bytes / (1024 ** 3)
                if free_vram_gb < 3.0:
                    max_chars = min(max_chars, 6000)   # Drastically clamp to prevent VRAM OOM
                elif free_vram_gb < 6.0:
                    max_chars = min(max_chars, 12000)  # Moderate clamping
        except Exception:
            pass

        # Check system RAM
        if psutil:
            try:
                free_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
                if free_ram_gb < 3.5:
                    max_chars = min(max_chars, 6000)
            except Exception:
                pass
        else:
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            free_mem_kb = int(line.split()[1])
                            free_ram_gb = free_mem_kb / (1024 ** 2)
                            if free_ram_gb < 3.5:
                                max_chars = min(max_chars, 6000)
                            break
            except Exception:
                pass

        file_entries = []
        
        # 1. Collect Python and JavaScript/TypeScript source files
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs and not d.startswith('.')]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_dir)
                file_entries.append((file_path, rel_path))
                
        # 2. Score files based on relevance to prompt
        scored_entries = []
        if prompt:
            prompt_lower = prompt.lower()
            for file_path, rel_path in file_entries:
                score = 0
                parts = rel_path.lower().split(os.sep)
                filename = parts[-1]
                
                # Check if exact filename is mentioned (high priority)
                if filename in prompt_lower:
                    score += 100
                # Check if parts of the path are mentioned (medium priority)
                for part in parts[:-1]:
                    if len(part) > 3 and part in prompt_lower:
                        score += 30
                        
                scored_entries.append((score, file_path, rel_path))
        else:
            scored_entries = [(0, fp, rp) for fp, rp in file_entries]
            
        # 3. Sort by score (descending) and alphabetically
        scored_entries.sort(key=lambda x: (-x[0], x[2]))
        
        # 4. Generate signatures up to clamped max_chars
        map_output = []
        current_len = 0
        truncated = False
        
        for score, file_path, rel_path in scored_entries:
            signatures = self._parse_file(file_path)
            if signatures:
                entry_header = f"# {rel_path}"
                entry_body = "\n".join(signatures) + "\n\n"
                entry_str = f"{entry_header}\n{entry_body}"
                
                if current_len + len(entry_str) > max_chars:
                    truncated = True
                    break
                    
                map_output.append(entry_str)
                current_len += len(entry_str)
                
        if truncated:
            map_output.append("# ... [other repository files truncated to fit context limits]")
            
        return "".join(map_output).strip()

if __name__ == "__main__":
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    repo_map = RepoMapGenerator(backend_dir).generate_map()
    print("=== AST REPO MAP FOR BACKEND ===")
    print(repo_map)
    print("================================")
