"""
Cache Management Utility
Provides functions to clear various caches used in the application.
"""
import os
import shutil
import glob
import logging

logger = logging.getLogger(__name__)

def clear_all_caches():
    """
    Deletes various cache files and directories from the project root.

    Returns:
        A dictionary summarizing the actions taken.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    report = {
        "files_deleted": [],
        "dirs_deleted": [],
        "errors": []
    }

    # --- Files to delete ---
    files_to_delete = [
        "news_cache.json",
        "signals_cache.json",
        "ticker_validation_cache.json",
        "coverage.xml"
    ]
    for filename in files_to_delete:
        path = os.path.join(project_root, filename)
        if os.path.exists(path):
            try:
                os.remove(path)
                report["files_deleted"].append(filename)
                logger.info(f"Deleted cache file: {filename}")
            except Exception as e:
                error_msg = f"Error deleting file {filename}: {e}"
                report["errors"].append(error_msg)
                logger.error(error_msg)

    # --- Directories to delete ---
    dirs_to_delete = [
        "data_cache",
        "intraday_cache",
        ".pytest_cache",
        "__pycache__",
        "src/__pycache__"
    ]
    for dirname in dirs_to_delete:
        path = os.path.join(project_root, dirname)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                report["dirs_deleted"].append(dirname)
                logger.info(f"Deleted cache directory: {dirname}")
            except Exception as e:
                error_msg = f"Error deleting directory {dirname}: {e}"
                report["errors"].append(error_msg)
                logger.error(error_msg)
    
    # --- Glob patterns for directories ---
    glob_dirs = glob.glob(os.path.join(project_root, '**', '__pycache__'), recursive=True)
    for path in glob_dirs:
        if os.path.exists(path) and path not in report["dirs_deleted"]:
             try:
                shutil.rmtree(path)
                relative_path = os.path.relpath(path, project_root)
                report["dirs_deleted"].append(relative_path)
                logger.info(f"Deleted cache directory: {relative_path}")
             except Exception as e:
                error_msg = f"Error deleting glob directory {path}: {e}"
                report["errors"].append(error_msg)
                logger.error(error_msg)

    return report
