import os
import sys
import shutil
import requests
import time
from huggingface_hub import hf_hub_download

# Default local models directory
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")))

# Definitions of our local 2B-8B LLMs and multimodal vision/OCR assistants.
# Categorized into 'text' and 'image_to_text' subfolders.
MODEL_DEFINITIONS = {
    'qwen_vl': {
        'repo_id': 'unsloth/Qwen2.5-VL-7B-Instruct-GGUF',
        'filename': 'Qwen2.5-VL-7B-Instruct-UD-Q6_K_XL.gguf',
        'mmproj_filename': 'mmproj-BF16.gguf',
        'name': 'Qwen-2.5-VL-7B (Vision/Doc Parsing)',
        'type': 'image_to_text',
    },
    'router': {
        'repo_id': 'bartowski/Phi-3.5-mini-instruct-GGUF',
        'filename': 'Phi-3.5-mini-instruct-Q6_K.gguf',
        'name': 'Phi-3.5-Mini (Master Router)',
        'type': 'text',
    },
    'deepseek_r1': {
        'repo_id': 'unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF',
        'filename': 'DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf',
        'name': 'DeepSeek-R1-7B (Reasoning Engine)',
        'type': 'text',
    },
    'opencode': {
        'repo_id': 'MaziyarPanahi/OpenCodeInterpreter-DS-6.7B-GGUF',
        'filename': 'OpenCodeInterpreter-DS-6.7B.Q6_K.gguf',
        'name': 'OpenCodeInterpreter 6.7B (3D Visualization Layer)',
        'type': 'text',
    }
}

def get_model_filenames(definition):
    filenames = []
    filename = definition["filename"]
    import re
    match = re.search(r'^(.*?)(\d+)-of-(\d+)(.*?)$', filename)
    if match:
        prefix, start, total, suffix = match.groups()
        total_shards = int(total)
        width = len(start)
        for i in range(1, total_shards + 1):
            shard_num = str(i).zfill(width)
            filenames.append(f"{prefix}{shard_num}-of-{total}{suffix}")
    else:
        filenames.append(filename)
    
    if "mmproj_filename" in definition:
        filenames.append(definition["mmproj_filename"])
    return filenames

def is_model_downloaded(model_key):
    if model_key not in MODEL_DEFINITIONS:
        return False
    definition = MODEL_DEFINITIONS[model_key]
    filenames = get_model_filenames(definition)
    subfolder = definition.get("type", "text")
    if "/" in subfolder:
        subfolder = subfolder.split("/")[-1]
    target_dir = os.path.join(MODELS_DIR, subfolder, model_key)
    
    # Check standard nested structure
    nested_exists = all(os.path.exists(os.path.join(target_dir, fname)) for fname in filenames)
    if nested_exists:
        return True
        
    # Check flat structure (common in Kaggle datasets)
    flat_exists = all(os.path.exists(os.path.join(MODELS_DIR, fname)) for fname in filenames)
    return flat_exists

def get_model_path(model_key):
    """Get the local path for a model key, downloading it if not present (returns the path to the first shard/file)."""
    if model_key not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model key: {model_key}")
        
    definition = MODEL_DEFINITIONS[model_key]
    subfolder = definition.get("type", "text")
    if "/" in subfolder:
        subfolder = subfolder.split("/")[-1]
        
    # Check if it exists in a flat structure (Kaggle datasets)
    flat_path = os.path.join(MODELS_DIR, definition["filename"])
    if os.path.exists(flat_path):
        return flat_path

    # Target directory under models/<subcategory>/<model_key>/
    target_dir = os.path.join(MODELS_DIR, subfolder, model_key)
    try:
        os.makedirs(target_dir, exist_ok=True)
    except OSError:
        pass # Handle read-only file systems (like mounted Kaggle Datasets)
    
    local_path = os.path.join(target_dir, definition["filename"])
    return local_path

def check_models_status():
    """Check which models are downloaded and ready."""
    status = {}
    for key, definition in MODEL_DEFINITIONS.items():
        subfolder = definition.get("type", "text")
        if "/" in subfolder:
            subfolder = subfolder.split("/")[-1]
        target_dir = os.path.join(MODELS_DIR, subfolder, key)
        
        filenames = get_model_filenames(definition)
        all_downloaded = True
        total_size = 0
        first_path = os.path.join(target_dir, filenames[0])
        
        for fname in filenames:
            path = os.path.join(target_dir, fname)
            if not os.path.exists(path):
                all_downloaded = False
            else:
                total_size += os.path.getsize(path)
                
        status[key] = {
            "name": definition["name"],
            "filename": definition["filename"],
            "repo_id": definition["repo_id"],
            "downloaded": all_downloaded,
            "path": first_path if all_downloaded else None,
            "size": f"{total_size / (1024**3):.2f} GB" if all_downloaded else "N/A"
        }
    return status

def download_model(model_key, progress_callback=None):
    """Download a model from Hugging Face (downloads all shards/files if sharded)."""
    if model_key not in MODEL_DEFINITIONS:
        raise ValueError(f"Unknown model key: {model_key}")
        
    definition = MODEL_DEFINITIONS[model_key]
    filenames = get_model_filenames(definition)
    
    subfolder = definition.get("type", "text")
    if "/" in subfolder:
        subfolder = subfolder.split("/")[-1]
    target_dir = os.path.join(MODELS_DIR, subfolder, model_key)
    os.makedirs(target_dir, exist_ok=True)
    
    for idx, fname in enumerate(filenames):
        local_path = os.path.join(target_dir, fname)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        temp_path = local_path + ".incomplete"
        
        # Check if already complete
        if os.path.exists(local_path):
            print(f"[{idx+1}/{len(filenames)}] {fname} already exists at {local_path}.")
            continue
            
        url = f"https://huggingface.co/{definition['repo_id']}/resolve/main/{fname}"
        print(f"[{idx+1}/{len(filenames)}] Downloading {fname} from {url}...")
        
        max_retries = 15
        for attempt in range(max_retries):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                try:
                    import huggingface_hub
                    token = huggingface_hub.get_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                except Exception:
                    pass
                
                initial_pos = 0
                if os.path.exists(temp_path):
                    initial_pos = os.path.getsize(temp_path)
                    headers["Range"] = f"bytes={initial_pos}-"
                    print(f"\nResuming download from {initial_pos / (1024**2):.2f} MB...")
                    
                response = requests.get(url, stream=True, headers=headers, timeout=60)
                
                if response.status_code == 416:
                    print("\nRange error, starting download from scratch...")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    initial_pos = 0
                    headers_scratch = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    }
                    if "Authorization" in headers:
                        headers_scratch["Authorization"] = headers["Authorization"]
                    response = requests.get(url, stream=True, headers=headers_scratch, timeout=60)
                    
                response.raise_for_status()
                
                mode = "ab" if (response.status_code == 206 and initial_pos > 0) else "wb"
                if mode == "wb":
                    initial_pos = 0
                    
                total_size = int(response.headers.get('content-length', 0)) + initial_pos
                print(f"Total Size: {total_size / (1024**2):.2f} MB")
                
                downloaded = initial_pos
                chunk_size = 1024 * 1024
                
                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB)")
                            sys.stdout.flush()
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                sys.stdout.write("\n")
                break
            except requests.exceptions.HTTPError as http_err:
                status_code = http_err.response.status_code if http_err.response is not None else None
                print(f"\nHTTP error occurred: {status_code} - {str(http_err)}")
                if status_code in [401, 403, 404]:
                    print("Permanent client error (unauthorized/forbidden/not found). Aborting retries.")
                    raise http_err
                if attempt < max_retries - 1:
                    sleep_time = 3 * (attempt + 1)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise http_err
            except Exception as e:
                print(f"\nDownload attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = 3 * (attempt + 1)
                    print(f"Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                else:
                    raise e
                    
        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(temp_path, local_path)
        print(f"Successfully downloaded {fname} to {local_path}")
        
    return os.path.join(target_dir, filenames[0])

def migrate_existing_models():
    """Migrate existing model files to their new flat subcategory model-specific folders."""
    if not os.path.exists(MODELS_DIR):
        return
        
    for key, definition in MODEL_DEFINITIONS.items():
        new_path = get_model_path(key)
        if os.path.exists(new_path):
            continue
            
        # 1. Candidate 1: directly under models/<subcategory>/filename (from the flat subcategory step)
        subfolder = definition.get("type", "text")
        if "/" in subfolder:
            subfolder = subfolder.split("/")[-1]
        old_path_flat = os.path.join(MODELS_DIR, subfolder, definition["filename"])
        if os.path.exists(old_path_flat) and os.path.isfile(old_path_flat) and old_path_flat != new_path:
            try:
                print(f"Migrating {definition['filename']} (flat) -> {new_path}...")
                shutil.move(old_path_flat, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")
                
        # 2. Candidate 2: old nested type path (e.g. models/natural_language_processing_nlp/sentence_similarity/file)
        old_nested_folder = os.path.join(MODELS_DIR, definition.get("type", "text"))
        old_path_nested = os.path.join(old_nested_folder, definition["filename"])
        if os.path.exists(old_path_nested) and os.path.isfile(old_path_nested) and old_path_nested != new_path:
            try:
                print(f"Migrating {definition['filename']} (nested) -> {new_path}...")
                shutil.move(old_path_nested, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")

        # 3. Candidate 3: root models/filename
        old_path_root = os.path.join(MODELS_DIR, definition["filename"])
        if os.path.exists(old_path_root) and os.path.isfile(old_path_root) and old_path_root != new_path:
            try:
                print(f"Migrating {definition['filename']} (root) -> {new_path}...")
                shutil.move(old_path_root, new_path)
            except Exception as e:
                print(f"Error migrating {definition['filename']}: {e}")

# Run model file categorization migration on load
migrate_existing_models()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Download specific models
        for arg in sys.argv[1:]:
            if arg in MODEL_DEFINITIONS:
                try:
                    download_model(arg)
                except Exception as e:
                    print(f"Error downloading {arg}: {str(e)}")
            else:
                print(f"Unknown model: {arg}")
    else:
        # Test script to check status
        print("Checking local models status...")
        status = check_models_status()
        for key, info in status.items():
            status_str = "✅ Ready" if info["downloaded"] else "❌ Missing"
            print(f"- {info['name']}: {status_str} ({info['filename']})")
