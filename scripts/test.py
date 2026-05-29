from pathlib import Path
import subprocess

def get_test_suite(dirpath: Path, suffix: str) -> tuple[list[Path], list[Path], list[Path]]:
    tests = [f for f in dirpath.rglob(f"**/*.{suffix}")]
    stdouts = [f for f in dirpath.rglob("**/*.stdout")]
    stderrs = [f for f in dirpath.rglob("**/*.stderr")]
    tests.sort()
    stdouts.sort()
    stderrs.sort()
    return tests, stdouts, stderrs

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
        print(f"Expected stdout:\n{expect_stdout.replace(" ", ".").replace("\n", "\n|")}")
        print(f"Actual stdout:\n{stdout.replace(" ", ".").replace("\n", "\n|")}")
        return False
    if stderr != expect_stderr:
        print(f"Error: {cmd}")
        print(f"Expected stderr:\n{expect_stderr.replace(" ", ".").replace("\n", "\n|")}")
        print(f"Actual stderr:\n{stderr.replace(" ", ".").replace("\n", "\n|")}")
        return False
    return True

def test_lexer(executable: Path) -> tuple[int, int]:
    lex_test, lex_stdout, lex_stderr = get_test_suite(Path("test/lexer"), "colgm")
    passed = 0
    for i in range(len(lex_test)):
        cmd = [str(executable), str(lex_test[i]), "--lex"]
        with open(lex_stdout[i], "r") as f:
            stdout = f.read()
        with open(lex_stderr[i], "r") as f:
            stderr = f.read()
        res = check(cmd, stdout, stderr)
        passed += 1 if res else 0
        if res:
            print("[PASSED]", lex_test[i])
        else:
            print("[FAILED]", lex_test[i])
    return passed, len(lex_test)

def test_colgm_opt(executable: Path) -> tuple[int, int]:
    colgm_opt_test, colgm_opt_stdout, colgm_opt_stderr = get_test_suite(Path("test/mlir"), "mlir")
    passed = 0
    for i in range(len(colgm_opt_test)):
        cmd = [str(executable), str(colgm_opt_test[i])]
        with open(colgm_opt_stdout[i], "r") as f:
            stdout = f.read()
        with open(colgm_opt_stderr[i], "r") as f:
            stderr = f.read()
        res = check(cmd, stdout, stderr)
        passed += 1 if res else 0
        if res:
            print("[PASSED]", colgm_opt_test[i])
        else:
            print("[FAILED]", colgm_opt_test[i])
    return passed, len(colgm_opt_test)

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

    lex_passed, len_lex_test = test_lexer(colgm_mlir)
    passed += lex_passed
    len_test += len_lex_test

    colgm_opt_passed, len_colgm_opt_test = test_colgm_opt(colgm_opt)
    passed += colgm_opt_passed
    len_test += len_colgm_opt_test
    
    print("=" * 60)
    print(f"{passed}/{len_test} passed")
    print("=" * 60)