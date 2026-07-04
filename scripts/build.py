import subprocess
from pathlib import Path
import os
import time

build_dir = Path("build")
cmakefiles_dir = Path("CMakeFiles")

def cpu_count():
    n = os.cpu_count()
    if n <= 1:
        return 1
    if n > 5:
        return n // 2
    return n - 1

def main():
    if not build_dir.exists():
        build_dir.mkdir()

    print("[INFO] chdir", build_dir.absolute())
    os.chdir(build_dir)

    # configure
    if not cmakefiles_dir.exists():
        cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=RelWithDebInfo", "-DENABLE_MLIR=ON"]
        print("[INFO] Running", " ".join(cmd))
        subprocess.run(cmd).check_returncode()
    # build
    t0 = time.perf_counter()
    
    cmd = ["cmake", "--build", ".", "-j", f"{cpu_count()}"]
    print("[INFO] Running", " ".join(cmd))
    subprocess.run(cmd).check_returncode()

    print("[INFO] chdir", Path("..").absolute().resolve())
    os.chdir("..")

    elapsed = time.perf_counter() - t0
    print(f"[INFO] Build finished in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
