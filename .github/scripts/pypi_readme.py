"""Rewrite README.md in place so that it renders on PyPI.

GitHub resolves relative paths against the repository and turns a bare
``user-attachments`` video URL into an inline player. PyPI does neither: relative
images and links break, and the video URL renders as plain text. This script,
run by the release workflow just before ``python -m build``:

* points relative image sources at ``raw.githubusercontent.com`` and relative
  link targets at ``github.com/<repo>/blob``, both pinned to ``REF`` (the release
  tag), so the PyPI page never changes after release;
* replaces a bare video URL that follows a ``<!-- pypi-video-thumbnail: PATH -->``
  marker with a thumbnail image linking to the repository page at ``REF``, where
  GitHub plays the video;
* leaves fenced code blocks untouched, and exits non-zero if a relative target
  survives or a rewritten target does not exist in the checkout.

Usage: python .github/scripts/pypi_readme.py REPO REF [README]
e.g.   python .github/scripts/pypi_readme.py wienkers/marEx v5.0.0
"""

import re
import sys
from pathlib import Path

ABSOLUTE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|//)", re.IGNORECASE)
MARKER = re.compile(r"^<!--\s*pypi-video-thumbnail:\s*(\S+)\s*-->\s*$")
VIDEO = re.compile(r"^(https://github\.com/user-attachments/assets/[0-9a-f-]+)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
IMAGE_EXT = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.IGNORECASE)

# Each pattern captures (prefix, target, suffix); group 2 is the link target.
HTML_SRC = re.compile(r"""(<img\b[^>]*?\bsrc=["'])([^"']+)(["'])""", re.IGNORECASE)
HTML_HREF = re.compile(r"""(<a\b[^>]*?\bhref=["'])([^"']+)(["'])""", re.IGNORECASE)
MD_IMAGE = re.compile(r"(!\[[^\]]*\]\(\s*<?)([^)\s>]+)(>?(?:\s+\"[^\"]*\")?\s*\))")
MD_LINK = re.compile(r"((?<!!)\[[^\]]*\]\(\s*<?)([^)\s>]+)(>?(?:\s+\"[^\"]*\")?\s*\))")
MD_REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*<?)(\S+?)(>?(?:\s+.*)?)$")


def rewrite(text: str, repo: str, ref: str, root: Path) -> tuple[str, list[str]]:
    """Return the rewritten text and a list of problems (empty when the result is safe for PyPI)."""
    raw = f"https://raw.githubusercontent.com/{repo}/{ref}/"
    blob = f"https://github.com/{repo}/blob/{ref}/"
    problems: list[str] = []

    def absolute(target: str, base: str, lineno: int) -> str:
        if ABSOLUTE.match(target):
            return target
        path = target.removeprefix("./")
        if not (root / path.split("#", 1)[0].split("?", 1)[0]).exists():
            problems.append(f"line {lineno}: {target!r} does not exist in the checkout")
        return base + path

    def sub(pattern: re.Pattern, base_for, line: str, lineno: int) -> str:
        return pattern.sub(lambda m: m.group(1) + absolute(m.group(2), base_for(m.group(2)), lineno) + m.group(3), line)

    lines = text.splitlines(keepends=True)
    out = []
    in_fence = False
    i = 0
    while i < len(lines):
        line, lineno = lines[i], i + 1
        if FENCE.match(line):
            in_fence = not in_fence
        if in_fence or FENCE.match(line):
            out.append(line)
            i += 1
            continue
        marker = MARKER.match(line)
        if marker:
            video = VIDEO.match(lines[i + 1]) if i + 1 < len(lines) else None
            if not video:
                problems.append(f"line {lineno}: marker is not followed by a user-attachments video URL")
                out.append(line)
                i += 1
                continue
            # The asset URL itself is not public (404 when logged out); GitHub's rendered README plays it.
            thumb = absolute(marker.group(1), raw, lineno)
            page = f"https://github.com/{repo}/tree/{ref}"
            out.append(f'<a href="{page}"><img src="{thumb}" alt="Video: play on GitHub" width="100%"></a>\n')
            i += 2
            continue
        line = sub(HTML_SRC, lambda t: raw, line, lineno)
        line = sub(HTML_HREF, lambda t: raw if IMAGE_EXT.search(t) else blob, line, lineno)
        line = sub(MD_IMAGE, lambda t: raw, line, lineno)
        line = sub(MD_LINK, lambda t: raw if IMAGE_EXT.search(t) else blob, line, lineno)
        line = sub(MD_REFDEF, lambda t: raw if IMAGE_EXT.search(t) else blob, line, lineno)
        out.append(line)
        i += 1

    result = "".join(out)
    # Independent re-scan: no relative target may survive outside code fences.
    in_fence = False
    for lineno, line in enumerate(result.splitlines(), 1):
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for pattern in (HTML_SRC, HTML_HREF, MD_IMAGE, MD_LINK, MD_REFDEF):
            for m in pattern.finditer(line):
                if not ABSOLUTE.match(m.group(2)):
                    problems.append(f"output line {lineno}: relative target {m.group(2)!r} survived")
    return result, problems


def main(argv: list[str]) -> int:
    """Rewrite the README named on the command line (default README.md)."""
    if len(argv) not in (2, 3):
        raise SystemExit(__doc__)
    repo, ref = argv[0], argv[1]
    path = Path(argv[2] if len(argv) == 3 else "README.md")
    after, problems = rewrite(path.read_text(encoding="utf-8"), repo, ref, Path.cwd())
    if problems:
        print("pypi_readme: refusing to write a README that would break on PyPI:", *problems, sep="\n  ")
        return 1
    path.write_text(after, encoding="utf-8")
    print(f"pypi_readme: rewrote {path} for {repo}@{ref}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
