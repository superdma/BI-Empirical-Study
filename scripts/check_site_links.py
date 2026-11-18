#!/usr/bin/env python3
"""Fail if the built site contains a broken internal link.

Usage: python3 scripts/check_site_links.py [site_dir]
"""

from __future__ import annotations

import os
import re
import sys
from urllib.parse import unquote, urlsplit

LINK_RE = re.compile(r"""(?:href|src)\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
SKIP_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:")


def site_base(site_root: str) -> str:
    """Return the path prefix from mkdocs.yml site_url, e.g. /BI-Empirical-Study."""
    config = os.path.join(os.path.dirname(os.path.abspath(site_root)), "mkdocs.yml")
    try:
        with open(config, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""
    match = re.search(r"^site_url:\s*[\"']?([^\"'\s]+)", text, re.MULTILINE)
    if not match:
        return ""
    return urlsplit(match.group(1)).path.rstrip("/")


def target_for(site_root: str, page: str, raw: str, base: str = "") -> str | None:
    raw = raw.strip()
    if not raw or raw.startswith("#") or raw.lower().startswith(SKIP_SCHEMES):
        return None

    path = unquote(urlsplit(raw).path)
    if not path:
        return None

    if path.startswith("/"):
        if base and (path == base or path.startswith(base + "/")):
            path = path[len(base):] or "/"
        full = os.path.join(site_root, path.lstrip("/"))
    else:
        full = os.path.join(os.path.dirname(page), path)

    full = os.path.normpath(full)
    if os.path.isdir(full):
        full = os.path.join(full, "index.html")
    return full


def main() -> int:
    site_root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "site")
    if not os.path.isdir(site_root):
        print("site directory not found: {}".format(site_root), file=sys.stderr)
        return 2

    base = site_base(site_root)
    broken = []
    checked = 0

    for root, _dirs, files in os.walk(site_root):
        for name in files:
            if not name.endswith(".html"):
                continue
            page = os.path.join(root, name)
            with open(page, encoding="utf-8", errors="replace") as handle:
                html = handle.read()
            for raw in LINK_RE.findall(html):
                target = target_for(site_root, page, raw, base)
                if target is None:
                    continue
                checked += 1
                if not os.path.exists(target):
                    broken.append(
                        (
                            os.path.relpath(page, site_root),
                            raw,
                            os.path.relpath(target, site_root),
                        )
                    )

    if broken:
        print("broken internal links: {}".format(len(broken)), file=sys.stderr)
        for page, raw, target in broken:
            print("  {} -> {} (missing {})".format(page, raw, target), file=sys.stderr)
        return 1

    print("internal links checked: {}; broken: 0".format(checked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
