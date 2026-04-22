#!/usr/bin/env python3
"""Download a Dropbox shared link (including shared folders) without an API token.

For Dropbox shared *folders*, Dropbox typically serves a ZIP when `dl=1`.

Examples:
  python download_dropbox_public.py "<dropbox-link>" --out data.zip
  python download_dropbox_public.py "<dropbox-link>" --out ./downloads --extract ./downloads/extracted

Notes:
- Requires: `pip install requests`
- Uses streaming download + follows redirects.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.parse
import zipfile


def _normalize_dropbox_dl_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
    query["dl"] = ["1"]
    new_query = urllib.parse.urlencode(query, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


def _filename_from_headers(headers: dict[str, str]) -> str | None:
    cd = headers.get("content-disposition") or headers.get("Content-Disposition")
    if not cd:
        return None

    # filename*=UTF-8''... (RFC 5987)
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd)
    if m:
        return urllib.parse.unquote(m.group(1)).strip().strip('"')

    m = re.search(r"filename=([^;]+)", cd)
    if m:
        return m.group(1).strip().strip('"')

    return None


def _safe_filename(name: str) -> str:
    name = name.strip().strip("\u200e\u200f")
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", " ", name)
    return name or "download"


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for u in units:
        if value < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(value)} {u}"
            return f"{value:.2f} {u}"
        value /= 1024.0
    return f"{n} B"


def _download_streaming(
    *,
    session,
    url: str,
    out_path: str,
    resume: bool,
    quiet: bool,
    timeout: int,
) -> str:
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dropbox-downloader/1.0",
    }

    mode = "wb"
    existing_size = 0
    if resume and os.path.exists(out_path):
        existing_size = os.path.getsize(out_path)
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

    with session.get(url, stream=True, allow_redirects=True, headers=headers, timeout=timeout) as r:
        if r.status_code not in (200, 206):
            raise RuntimeError(f"HTTP {r.status_code} when downloading. Final URL: {r.url}")

        total = r.headers.get("Content-Length")
        total_bytes = int(total) + existing_size if total and r.status_code == 206 else int(total) if total else None

        if not quiet:
            final_name = _filename_from_headers(dict(r.headers))
            if final_name:
                print(f"Remote filename: {final_name}")
            if total_bytes:
                print(f"Total size: {_format_bytes(total_bytes)}")

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        downloaded = existing_size
        last_print = 0.0
        start = time.time()

        with open(out_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if quiet:
                    continue

                # Print every ~2 seconds or on completion
                if now - last_print >= 2.0:
                    last_print = now
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        elapsed = max(now - start, 1e-6)
                        speed = downloaded / elapsed
                        print(
                            f"Downloaded: {_format_bytes(downloaded)} / {_format_bytes(total_bytes)} "
                            f"({pct:.1f}%) at {_format_bytes(int(speed))}/s",
                            flush=True,
                        )
                    else:
                        print(f"Downloaded: {_format_bytes(downloaded)}", flush=True)

        if not quiet:
            print(f"Saved to: {out_path}")

    return out_path


def _resolve_out_path(out: str, filename_hint: str | None) -> str:
    if os.path.isdir(out) or out.endswith(os.sep):
        name = _safe_filename(filename_hint or "dropbox_download.zip")
        return os.path.join(out, name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Dropbox shared link without an API token")
    parser.add_argument("url", help="Dropbox shared link")
    parser.add_argument(
        "--out",
        default="./dropbox_download.zip",
        help="Output file path OR output directory (default: ./dropbox_download.zip)",
    )
    parser.add_argument(
        "--extract",
        nargs="?",
        const=".",
        default=None,
        help="If the download is a zip, extract it to this directory (default if flag provided: current directory)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing partial download if possible")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds (default: 60)")

    args = parser.parse_args()

    try:
        import requests  # type: ignore
    except Exception:
        print("Missing dependency: requests. Install with: pip install requests", file=sys.stderr)
        return 2

    dl_url = _normalize_dropbox_dl_url(args.url)

    with requests.Session() as session:
        # A HEAD preflight to capture filename, but Dropbox sometimes blocks HEAD.
        filename_hint = None
        try:
            head = session.head(
                dl_url,
                allow_redirects=True,
                timeout=min(args.timeout, 30),
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) dropbox-downloader/1.0"},
            )
            if head.status_code in (200, 302, 301):
                filename_hint = _filename_from_headers(dict(head.headers))
        except Exception:
            pass

        out_path = _resolve_out_path(args.out, filename_hint)

        downloaded_path = _download_streaming(
            session=session,
            url=dl_url,
            out_path=out_path,
            resume=args.resume,
            quiet=args.quiet,
            timeout=args.timeout,
        )

    if args.extract is not None:
        extract_dir = args.extract
        if zipfile.is_zipfile(downloaded_path):
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(downloaded_path, "r") as zf:
                zf.extractall(extract_dir)
            if not args.quiet:
                print(f"Extracted to: {extract_dir}")
        else:
            print("--extract was set, but the downloaded file is not a zip; skipping extraction.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
