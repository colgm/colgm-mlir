from pathlib import Path
import subprocess
import sys

def find_binary_tests():
    binary_tests = [
        "build/type_storage_test"
    ]
    res = []
    for i in binary_tests:
        if not Path(i).exists():
            continue
        res.append(Path(i))
    return res

def test_binary() -> tuple[int, int]:
    tests = find_binary_tests()
    passed = 0
    for t in tests:
        res = subprocess.run([str(t)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("[FAILED]", t)
        else:
            print("[PASSED]", t)
            passed += 1
    return passed, len(tests)

def get_test_suite(dirpath: Path, suffix: str) -> list[Path]:
    tests = [f for f in dirpath.rglob(f"**/*.{suffix}")]
    tests.sort()
    return tests

def de_color(text: str) -> str:
    color = [
        "0", "0;1", "1", "7", "90", "90;1",
        "91", "91;1", "92", "92;1",
        "93", "93;1", "94", "94;1",
        "95", "95;1", "96", "96;1",
        "97", "97;1", "36", "36;1"
    ]
    for i in color:
        text = text.replace("\033[" + i + "m", "")
    text = text.replace("\r", "")
    return text

def check(cmd: list[str], expect_stdout: str, expect_stderr: str) -> bool:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expect_stdout = de_color(expect_stdout)
    expect_stderr = de_color(expect_stderr)
    stdout = de_color(res.stdout.decode("utf-8"))
    stderr = de_color(res.stderr.decode("utf-8"))
    if stdout != expect_stdout:
        print(f"Error: {cmd}")
        eo = expect_stdout.replace(" ", ".").replace("\n", "\n|")
        ao = stdout.replace(" ", ".").replace("\n", "\n|")
        print(f"Expected stdout:\n{eo}")
        print(f"Actual stdout:\n{ao}")
        return False
    if stderr != expect_stderr:
        print(f"Error: {cmd}")
        ee = expect_stderr.replace(" ", ".").replace("\n", "\n|")
        ae = stderr.replace(" ", ".").replace("\n", "\n|")
        print(f"Expected stderr:\n{ee}")
        print(f"Actual stderr:\n{ae}")
        return False
    return True

def test(executable: Path, directory: Path, suffix: str, options: list[str]) -> tuple[int, int]:
    stage_test = get_test_suite(directory, suffix)
    passed = 0
    for i in range(len(stage_test)):
        test_file = stage_test[i]
        test_file_stdout = Path(str(test_file) + ".stdout")
        test_file_stderr = Path(str(test_file) + ".stderr")
        if not test_file_stdout.exists() or not test_file_stderr.exists():
            print(f"[FAILED] cannot find stdout/stderr file of {test_file}")
            continue
        cmd = [str(executable), str(test_file)] + options
        with open(test_file_stdout, "r") as f:
            stdout = f.read()
        with open(test_file_stderr, "r") as f:
            stderr = f.read()
        res = check(cmd, stdout, stderr)
        passed += 1 if res else 0
        if res:
            print("[PASSED]", stage_test[i])
        else:
            print("[FAILED]", stage_test[i])
    return passed, len(stage_test)

def test_lexer(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/lexer"), "colgm", ["--lex"])

def test_parser(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/parse"), "colgm", ["--ast"])

def test_sema(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/sema"), "colgm", ["--sema"])

def test_codegen(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/codegen"), "colgm", ["--mlir"])
def test_colgm_opt(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/mlir"), "mlir", [])

if __name__ == "__main__":
    colgm_mlir = Path("build/colgm-mlir")
    colgm_opt = Path("build/colgm-opt")
    if not colgm_mlir.exists():
        print("Error: executable not found")
        exit(1)
    if not colgm_opt.exists():
        print("Error: executable not found")
        exit(1)

    passed, len_test = 0, 0

    print("=" * 60)
    print("Testing colgm-mlir::lexer")
    print("=" * 60)
    lex_passed, len_lex_test = test_lexer(colgm_mlir)
    passed += lex_passed
    len_test += len_lex_test

    print("=" * 60)
    print("Testing colgm-mlir::parser")
    print("=" * 60)
    parser_passed, len_parser_test = test_parser(colgm_mlir)
    passed += parser_passed
    len_test += len_parser_test

    print("=" * 60)
    print("Testing colgm-mlir::sema")
    print("=" * 60)
    sema_passed, len_sema_test = test_sema(colgm_mlir)
    passed += sema_passed
    len_test += len_sema_test

    print("=" * 60)
    print("Testing colgm-mlir::codegen")
    print("=" * 60)
    codegen_passed, len_codegen_test = test_codegen(colgm_mlir)
    passed += codegen_passed
    len_test += len_codegen_test

    print("=" * 60)
    print("Testing colgm-opt")
    print("=" * 60)
    colgm_opt_passed, len_colgm_opt_test = test_colgm_opt(colgm_opt)
    passed += colgm_opt_passed
    len_test += len_colgm_opt_test

    print("=" * 60)
    print("Testing modules functionality")
    print("=" * 60)
    bin_passed, len_bin_test = test_binary()
    passed += bin_passed
    len_test += len_bin_test

    print("=" * 60)
    print(f"{passed}/{len_test} passed")
    print("=" * 60)

    if passed != len_test:
        sys.exit(1)
