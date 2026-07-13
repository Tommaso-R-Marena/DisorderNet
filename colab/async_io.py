"""
I/O concurrency helpers (accuracy-preserving).

Uses threads / concurrent.futures for network-bound work. Not for GPU training.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

import requests


def _fetch_disprot_page(
    base_url: str,
    page: int,
    per_page: int = 100,
    timeout: int = 60,
    max_retries: int = 4,
) -> dict:
    params = {
        "release": "current",
        "show_ambiguous": "false",
        "show_obsolete": "false",
        "format": "json",
        "page": page,
        "per_page": per_page,
    }
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(base_url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"DisProt page {page} failed: {last_exc}")


def fetch_disprot_concurrent(
    cache_path: str = "disprot_raw.json",
    max_workers: int = 8,
    per_page: int = 100,
) -> list:
    """
    Concurrent DisProt REST download (page 0 → total, then parallel remaining pages).

    Same JSON schema as sequential fetch_disprot — byte-identical entry sets when
    API is stable (order sorted by page for determinism).
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "data" in payload:
            meta = payload.get("meta", {})
            print(
                f"Loading cached DisProt from '{cache_path}'...  "
                f"({meta.get('n_entries', len(payload['data']))} entries)",
                flush=True,
            )
            return payload["data"]
        return payload

    base_url = "https://disprot.org/api/search"
    print(f"Downloading DisProt (concurrent, workers={max_workers})…", flush=True)
    t0 = time.time()

    first = _fetch_disprot_page(base_url, page=0, per_page=per_page)
    page0 = first.get("data", [])
    total = int(first.get("total", len(page0)))
    n_pages = max(1, (total + per_page - 1) // per_page)

    pages: dict[int, list] = {0: page0}
    if n_pages > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_fetch_disprot_page, base_url, p, per_page): p
                for p in range(1, n_pages)
            }
            for fut in as_completed(futs):
                p = futs[fut]
                data = fut.result()
                pages[p] = data.get("data", [])
                print(f"  page {p + 1}/{n_pages}  (+{len(pages[p])} entries)", flush=True)

    all_entries: list = []
    for p in range(n_pages):
        all_entries.extend(pages.get(p, []))

    # Trim to declared total if API returned extras on last page
    if total and len(all_entries) > total:
        all_entries = all_entries[:total]

    elapsed = time.time() - t0
    print(f"Downloaded {len(all_entries)} entries in {elapsed:.0f}s (concurrent)", flush=True)

    meta = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_entries": len(all_entries),
        "api_release": "current",
        "content_sha256": hashlib.sha256(
            json.dumps(all_entries, sort_keys=True).encode()
        ).hexdigest(),
        "fetch_mode": "concurrent",
        "max_workers": max_workers,
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"meta": meta, "data": all_entries}, f)
    print(f"  Saved DisProt cache (sha256={meta['content_sha256'][:12]}...)")
    return all_entries


def run_overlapped(
    primary: Callable[[], Any],
    background: Callable[[], Any],
    primary_name: str = "primary",
    background_name: str = "background",
) -> tuple[Any, Any]:
    """
    Run primary on the calling thread while background runs in a worker thread.

    Typical: primary=load ESM, background=prefetch AF pLDDT.
    """
    print(f"Overlap: {background_name} ‖ {primary_name}", flush=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        bg_fut: Future = pool.submit(background)
        primary_result = primary()
        background_result = bg_fut.result()
    return primary_result, background_result


def mirror_files_parallel(
    paths: list[str],
    dest_dir: str,
    max_workers: int = 8,
) -> list[str]:
    """Copy existing files to dest_dir concurrently."""
    os.makedirs(dest_dir, exist_ok=True)
    import shutil

    def _one(src: str) -> Optional[str]:
        if not src or not os.path.isfile(src):
            return None
        dest = os.path.join(dest_dir, os.path.basename(src))
        shutil.copy2(src, dest)
        return dest

    copied: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for result in pool.map(_one, paths):
            if result:
                copied.append(result)
    return copied
