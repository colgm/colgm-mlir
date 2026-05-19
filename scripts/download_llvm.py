#!/usr/bin/env python3
"""
Download precompiled LLVM/MLIR binary package.

Primary mirror (fast in China):
  Tsinghua TUNA: https://mirrors.tuna.tsinghua.edu.cn/github-release/llvm/llvm-project/

Fallback:
  GitHub: https://github.com/llvm/llvm-project/releases/

Usage:
  python3 download_llvm.py                    # download latest (default 22.1.6)
  python3 download_llvm.py --version 19.1.0   # specific version
  python3 download_llvm.py --arch ARM64       # ARM64 architecture
  python3 download_llvm.py --output /opt/llvm # custom output dir

Supported platforms: Linux-X64 (default), Linux-ARM64
Supported versions: any LLVM release tag (e.g., 22.1.6, 19.1.0, 18.1.8)
"""

import argparse
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

# ---- config ----

DEFAULT_VERSION = "22.1.6"
DEFAULT_PLATFORM = "Linux-X64"

# Mirrors ordered by preference (for users in China, Tsinghua first)
MIRRORS: list[tuple[str, str]] = [
    (
        "Tsinghua TUNA",
        "https://mirrors.tuna.tsinghua.edu.cn/github-release/llvm/llvm-project/LLVM%20{version}/{filename}",
    ),
    (
        "GitHub",
        "https://github.com/llvm/llvm-project/releases/download/llvmorg-{version}/{filename}",
    ),
]

# ---- helpers ----

def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


class ProgressBar:
    """Simple progress bar without external dependencies."""
    def __init__(self, total: int, label: str = "", width: int = 40):
        self.total = total
        self.label = label
        self.width = width
        self.current = 0
        self.start_time = time.time()
        self._last_print = 0

    def update(self, n: int):
        self.current += n
        now = time.time()
        if now - self._last_print < 0.1 and self.current < self.total:
            return
        self._last_print = now
        self._render()

    def _render(self):
        pct = self.current / self.total if self.total else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0
        line = (
            f"\r  {self.label}[{bar}] {pct*100:5.1f}%  "
            f"{format_size(self.current)}/{format_size(self.total)}  "
            f"{format_size(speed)}/s  ETA: {format_duration(eta)}"
        )
        sys.stderr.write(line.ljust(120))
        sys.stderr.flush()

    def finish(self):
        self.current = self.total
        self._render()
        sys.stderr.write("\n")


def download_with_progress(url: str, dest: Path, label: str = "") -> bool:
    """Download a file with progress bar. Returns True on success."""
    try:
        req = Request(url, headers={"User-Agent": "colgm-downloader/1.0"})
        with urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            bar = ProgressBar(total, label=label) if total else None

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    if bar:
                        bar.update(len(chunk))
            if bar:
                bar.finish()
        return True
    except URLError as e:
        sys.stderr.write(f"\n  Download failed: {e}\n")
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        return False


def extract_tar_xz(archive: Path, dest_dir: Path) -> bool:
    """Extract .tar.xz archive. Requires xz utils or Python's tarfile."""
    print(f"  Extracting to {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Use system tar for better performance on large files
        subprocess.run(
            ["tar", "-xJf", str(archive), "-C", str(dest_dir)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  (falling back to Python tarfile, this may be slower)")
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(path=dest_dir, filter="fully_trusted")
    return True


# ---- main ----

def main():
    ap = argparse.ArgumentParser(
        description="Download precompiled LLVM/MLIR binary package"
    )
    ap.add_argument(
        "--version", "-v",
        default=DEFAULT_VERSION,
        help=f"LLVM version tag (default: {DEFAULT_VERSION})",
    )
    ap.add_argument(
        "--arch", "-a",
        default=DEFAULT_PLATFORM,
        help=f"Target platform (default: {DEFAULT_PLATFORM})",
    )
    ap.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: ./llvm-{version}-{arch}/)",
    )
    ap.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction, keep only the tarball",
    )
    ap.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive after extraction",
    )
    args = ap.parse_args()

    version = args.version
    platform = args.arch
    base_name = f"LLVM-{version}-{platform}"
    filename = f"{base_name}.tar.xz"

    # Determine output directory
    if args.output:
        extract_dir = Path(args.output)
    else:
        extract_dir = Path.cwd() / base_name

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Output directory already exists and is not empty: {extract_dir}")
        print(f"  Skipping download.")
        return

    # Build mirror URLs
    urls: list[tuple[str, str]] = []
    for mirror_name, template in MIRRORS:
        url = template.format(version=version, filename=filename)
        urls.append((mirror_name, url))

    print(f"  Target: LLVM {version} for {platform}")

    # Try each mirror
    downloaded = False
    archive_path = extract_dir.parent / filename  # Save next to extract dir

    for mirror_name, url in urls:
        print(f"  Trying {mirror_name}: {url}")
        if download_with_progress(url, archive_path, label=f"[{mirror_name}] "):
            downloaded = True
            print(f"  ✓ Downloaded from {mirror_name}")
            break
        print(f"  ✗ {mirror_name} failed, trying next mirror...\n")

    if not downloaded:
        print("  ERROR: All mirrors failed.", file=sys.stderr)
        sys.exit(1)

    # Extract
    if not args.no_extract:
        extract_tar_xz(archive_path, extract_dir)
        print(f"  ✓ Extracted to {extract_dir.resolve()}")

        # Verify MLIR is present
        mlir_cmake = extract_dir / base_name / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake"
        if mlir_cmake.exists():
            print(f"  ✓ MLIR development files detected")
        else:
            print(f"  ⚠ Warning: MLIRConfig.cmake not found at expected path")
            print(f"    Expected: {mlir_cmake}")

    # Cleanup
    if not args.keep_archive and not args.no_extract:
        archive_path.unlink()
        print(f"  ✓ Removed archive (use --keep-archive to preserve)")
    elif args.no_extract:
        print(f"  ✓ Archive saved at {archive_path.resolve()}")
    else:
        print(f"  ✓ Archive kept at {archive_path.resolve()}")

    print(f"\n  Done. Use with CMake:")
    print(f"    cmake -DLLVM_DIR={extract_dir.resolve()}/{base_name}/lib/cmake/llvm \\")
    print(f"          -DMLIR_DIR={extract_dir.resolve()}/{base_name}/lib/cmake/mlir \\")
    print(f"          ...")


if __name__ == "__main__":
    main()
