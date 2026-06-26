import subprocess
from pathlib import Path
import os

build_dir = Path("build")
cmakefiles_dir = Path("CMakeFiles")

def cpu_count():
    n = os.cpu_count()
    if n <= 1:
        return 1
    if n > 8:
        return n // 2
    return n - 1

def main():
    if not build_dir.exists():
        build_dir.mkdir()

    os.chdir(build_dir)
    # configure
    if not cmakefiles_dir.exists():
        subprocess.run(
            ["cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DENABLE_MLIR=ON"]
        ).check_returncode()
    # build
    subprocess.run(
        ["cmake", "--build", ".", "-j", f"{cpu_count()}"]
    ).check_returncode()
    os.chdir("..")

if __name__ == "__main__":
    main()
