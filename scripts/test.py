from pathlib import Path
import subprocess
import sys
import os
import difflib

def red(s: str) -> str:
    if not os.isatty(1):
        return s
    return "\033[91m" + s + "\033[0m"

def green(s: str) -> str:
    if not os.isatty(1):
        return s
    return "\033[92m" + s + "\033[0m"

def yellow(s: str) -> str:
    if not os.isatty(1):
        return s
    return "\033[93m" + s + "\033[0m"

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
            print(red("[FAILED]"), t)
        else:
            print(green("[PASSED]"), t)
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

def show_diff(expect: str, actual: str):
    diff = difflib.ndiff(
        expect.splitlines(),
        actual.splitlines()
    )

    print(yellow("==================== actual vs expected ===================="))
    for line in diff:
        if line.startswith("+"):
            print(green(line))
        elif line.startswith("-"):
            print(red(line))
        elif line.startswith("?"):
            print(yellow(line))
        else:
            print(line)
    print(yellow("========================= end diff ========================="))

def check(cmd: list[str], expect_stdout: str, expect_stderr: str) -> bool:
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    expect_stdout = de_color(expect_stdout)
    expect_stderr = de_color(expect_stderr)
    stdout = de_color(res.stdout.decode("utf-8"))
    stderr = de_color(res.stderr.decode("utf-8"))
    command = " ".join(cmd)
    if stdout != expect_stdout:
        print(f"{red('[ERROR-stdout]')}: {command}")
        show_diff(expect_stdout, stdout)
        return False
    if stderr != expect_stderr:
        print(f"{red('[ERROR-stderr]')}: {command}")
        show_diff(expect_stderr, stderr)
        return False
    return True

def test(executable: Path, directory: Path, suffix: str, options: list[str]) -> tuple[int, int]:
    stage_test = get_test_suite(directory, suffix)
    passed = 0
    for i in range(len(stage_test)):
        test_file = stage_test[i]
        test_file_stdout = Path(str(test_file) + ".stdout")
        test_file_stderr = Path(str(test_file) + ".stderr")
        option_file = Path(str(test_file) + ".option")

        # we see not exist stdout/stderr as empty input
        # but if both of them are not exist, we consider it invalid test
        if not test_file_stdout.exists() and not test_file_stderr.exists():
            print(f"[FAILED] cannot find stdout/stderr file of {test_file}")
            continue

        cmd = [str(executable)]
        if option_file.exists():
            with open(option_file, "r") as f:
                cmd += f.read().split(" ")
        cmd += [str(test_file)] + options

        # expect stdout
        if test_file_stdout.exists():
            with open(test_file_stdout, "r") as f:
                stdout = f.read()
        else:
            stdout = ""
        # expect stderr
        if test_file_stderr.exists():
            with open(test_file_stderr, "r") as f:
                stderr = f.read()
        else:
            stderr = ""

        res = check(cmd, stdout, stderr)
        passed += 1 if res else 0
        if res:
            print(green("[PASSED]"), stage_test[i])
        else:
            print(red("[FAILED]"), stage_test[i])
    return passed, len(stage_test)

def test_lexer(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/lexer"), "colgm", ["--lex"])

def test_parser(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/parse"), "colgm", ["--ast"])

def test_sema(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/sema"), "colgm", ["--sema"])

def test_codegen(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/codegen"), "colgm", ["--mlir"])

def test_lowering(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/lowering"), "colgm", ["--mlir"])

def test_jit(executable: Path) -> tuple[int, int]:
    return test(executable, Path("test/jit"), "colgm", ["--jit"])

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
    print("Testing colgm-mlir::lowering")
    print("=" * 60)
    lowering_passed, len_lowering_test = test_lowering(colgm_mlir)
    passed += lowering_passed
    len_test += len_lowering_test

    print("=" * 60)
    print("Testing colgm-mlir::jit")
    print("=" * 60)
    jit_passed, len_jit_test = test_jit(colgm_mlir)
    passed += jit_passed
    len_test += len_jit_test

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
    f = green if passed == len_test else yellow
    f = red if passed == 0 else f
    res = f(f"{passed}/{len_test}")
    print(f"{res} passed")
    print("=" * 60)

    if passed != len_test:
        sys.exit(1)
