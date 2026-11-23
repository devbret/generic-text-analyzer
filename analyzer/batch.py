import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import INPUT_DIR
from .pipeline import analyze_text

logger = logging.getLogger(__name__)


def find_text_files(input_dir: str) -> list[str]:
    return [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith(".txt")]


def analyze_files(paths: list[str], max_workers: int | None = None) -> list[dict]:
    if max_workers is None:
        max_workers = min(4, os.cpu_count() or 1)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_text, path) for path in paths]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logger.error("Error processing a file in parallel: %s", e)
    return results


def run(input_dir: str = INPUT_DIR) -> int:
    if not os.path.isdir(input_dir):
        logger.error("Input directory '%s' not found.", input_dir)
        return 1

    txt_files = find_text_files(input_dir)
    if not txt_files:
        logger.warning("No .txt files found in '%s'.", input_dir)
        return 0

    analyze_files(txt_files)
    return 0
