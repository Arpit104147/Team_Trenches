"""
git_agent.py — Agentic Git & Pull Request Automation for DeepThink AIOS
=========================================================================
Provides automated Git operations for the orchestrator pipeline:
    - Clone or pull repositories
    - Create feature branches
    - Commit generated files (Verilog, testbenches, layouts)
    - Push and create Pull Requests via GitHub API

Requires: gitpython>=3.1.0 (optional — graceful degradation if missing)
"""

import os
import json
import subprocess
import shutil
import tempfile

# Try to import GitPython for advanced Git operations
try:
    import git as gitpython
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

# Try to import requests for GitHub API
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class GitAgent:
    """Automated Git workspace management for the AIOS pipeline.

    Handles repository cloning, branch creation, file staging,
    commits, and optional GitHub Pull Request creation.
    """

    def __init__(self, workspace_base=None, github_token=None):
        """Initialize the Git Agent.

        Args:
            workspace_base: Base directory for git workspaces (default: ./workspaces/git)
            github_token: GitHub Personal Access Token for PR creation (optional)
        """
        self.workspace_base = workspace_base or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "workspaces", "git"
        )
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN", "")
        os.makedirs(self.workspace_base, exist_ok=True)

        self.git_binary = shutil.which("git")
        if not self.git_binary:
            print("GitAgent: ⚠️ git binary not found. Git operations disabled.")

    @property
    def available(self):
        """Check if git operations are available."""
        return self.git_binary is not None

    def clone_or_pull(self, repo_url, workspace_name=None):
        """Clone a repository or pull latest changes if already cloned.

        Args:
            repo_url: Git repository URL (HTTPS or SSH)
            workspace_name: Optional name for the workspace directory

        Returns:
            dict with keys: 'success', 'path', 'message'
        """
        if not self.available:
            return {"success": False, "path": None, "message": "Git not available"}

        # Derive workspace name from repo URL
        if not workspace_name:
            workspace_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")

        workspace_path = os.path.join(self.workspace_base, workspace_name)

        try:
            if os.path.exists(os.path.join(workspace_path, ".git")):
                # Pull latest changes
                result = subprocess.run(
                    [self.git_binary, "pull", "--rebase"],
                    cwd=workspace_path,
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    return {
                        "success": True,
                        "path": workspace_path,
                        "message": f"Updated existing workspace: {workspace_name}"
                    }
                else:
                    return {
                        "success": False,
                        "path": workspace_path,
                        "message": f"Pull failed: {result.stderr.strip()}"
                    }
            else:
                # Clone repository
                result = subprocess.run(
                    [self.git_binary, "clone", repo_url, workspace_path],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    return {
                        "success": True,
                        "path": workspace_path,
                        "message": f"Cloned repository to: {workspace_path}"
                    }
                else:
                    return {
                        "success": False,
                        "path": None,
                        "message": f"Clone failed: {result.stderr.strip()}"
                    }
        except subprocess.TimeoutExpired:
            return {"success": False, "path": None, "message": "Git operation timed out"}
        except Exception as e:
            return {"success": False, "path": None, "message": f"Git error: {str(e)}"}

    def create_branch(self, workspace_path, branch_name):
        """Create and checkout a new feature branch.

        Args:
            workspace_path: Path to the git repository
            branch_name: Name for the new branch

        Returns:
            dict with keys: 'success', 'message'
        """
        if not self.available:
            return {"success": False, "message": "Git not available"}

        try:
            # Create and switch to branch
            result = subprocess.run(
                [self.git_binary, "checkout", "-b", branch_name],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0:
                return {"success": True, "message": f"Created branch: {branch_name}"}
            else:
                # Branch might already exist, try switching
                result = subprocess.run(
                    [self.git_binary, "checkout", branch_name],
                    cwd=workspace_path,
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0:
                    return {"success": True, "message": f"Switched to existing branch: {branch_name}"}
                return {"success": False, "message": f"Branch creation failed: {result.stderr.strip()}"}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def commit_files(self, workspace_path, files_dict, commit_message):
        """Write files to the workspace and create a commit.

        Args:
            workspace_path: Path to the git repository
            files_dict: Dict mapping relative file paths to content strings
            commit_message: Git commit message

        Returns:
            dict with keys: 'success', 'message', 'sha'
        """
        if not self.available:
            return {"success": False, "message": "Git not available", "sha": None}

        try:
            # Write files
            for rel_path, content in files_dict.items():
                full_path = os.path.join(workspace_path, rel_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)

            # Stage all new/modified files
            subprocess.run(
                [self.git_binary, "add", "-A"],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=15
            )

            # Commit
            result = subprocess.run(
                [self.git_binary, "commit", "-m", commit_message],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=15
            )

            if result.returncode == 0:
                # Get commit SHA
                sha_result = subprocess.run(
                    [self.git_binary, "rev-parse", "HEAD"],
                    cwd=workspace_path,
                    capture_output=True, text=True, timeout=5
                )
                sha = sha_result.stdout.strip() if sha_result.returncode == 0 else "unknown"
                return {
                    "success": True,
                    "message": f"Committed {len(files_dict)} file(s): {commit_message}",
                    "sha": sha
                }
            else:
                return {
                    "success": False,
                    "message": f"Commit failed: {result.stderr.strip()}",
                    "sha": None
                }
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}", "sha": None}

    def push_branch(self, workspace_path, branch_name, remote="origin"):
        """Push a branch to the remote repository.

        Args:
            workspace_path: Path to the git repository
            branch_name: Name of the branch to push
            remote: Remote name (default: 'origin')

        Returns:
            dict with keys: 'success', 'message'
        """
        if not self.available:
            return {"success": False, "message": "Git not available"}

        try:
            result = subprocess.run(
                [self.git_binary, "push", "-u", remote, branch_name],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return {"success": True, "message": f"Pushed {branch_name} to {remote}"}
            else:
                return {"success": False, "message": f"Push failed: {result.stderr.strip()}"}
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Push timed out"}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def create_pull_request(self, repo_owner, repo_name, branch_name,
                            title, body, base_branch="main"):
        """Create a Pull Request on GitHub using the API.

        Args:
            repo_owner: GitHub repository owner
            repo_name: GitHub repository name
            branch_name: Source branch for the PR
            title: PR title
            body: PR description (markdown)
            base_branch: Target branch (default: 'main')

        Returns:
            dict with keys: 'success', 'message', 'url'
        """
        if not self.github_token:
            return {
                "success": False,
                "message": "GitHub token not configured. Set GITHUB_TOKEN env var or configure in settings.",
                "url": None
            }

        if not REQUESTS_AVAILABLE:
            return {
                "success": False,
                "message": "requests library not available for GitHub API calls.",
                "url": None
            }

        try:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json",
            }
            data = {
                "title": title,
                "body": body,
                "head": branch_name,
                "base": base_branch,
            }

            response = requests.post(url, headers=headers, json=data, timeout=30)

            if response.status_code == 201:
                pr_data = response.json()
                return {
                    "success": True,
                    "message": f"Pull Request #{pr_data['number']} created successfully",
                    "url": pr_data.get("html_url", "")
                }
            else:
                error_msg = response.json().get("message", response.text)
                return {
                    "success": False,
                    "message": f"GitHub API error ({response.status_code}): {error_msg}",
                    "url": None
                }
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}", "url": None}

    def get_workspace_status(self, workspace_path):
        """Get the current git status of a workspace.

        Args:
            workspace_path: Path to the git repository

        Returns:
            dict with workspace git information
        """
        if not self.available or not os.path.exists(os.path.join(workspace_path, ".git")):
            return {"valid": False, "message": "Not a git repository"}

        try:
            # Get current branch
            branch_result = subprocess.run(
                [self.git_binary, "branch", "--show-current"],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=5
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"

            # Get status
            status_result = subprocess.run(
                [self.git_binary, "status", "--porcelain"],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=5
            )
            changes = status_result.stdout.strip().split("\n") if status_result.stdout.strip() else []

            # Get latest commit
            log_result = subprocess.run(
                [self.git_binary, "log", "-1", "--oneline"],
                cwd=workspace_path,
                capture_output=True, text=True, timeout=5
            )
            latest_commit = log_result.stdout.strip() if log_result.returncode == 0 else "none"

            return {
                "valid": True,
                "branch": current_branch,
                "uncommitted_changes": len(changes),
                "latest_commit": latest_commit,
                "path": workspace_path,
            }
        except Exception as e:
            return {"valid": False, "message": f"Error: {str(e)}"}
