from pathlib import Path
import subprocess

def get_test_suite(dirpath: Path) -> tuple[list[Path], list[Path], list[Path]]:
    tests = [f for f in dirpath.rglob("**/*.colgm")]
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

if __name__ == "__main__":
    executable = Path("build/colgm-mlir")
    if not executable.exists():
        print("Error: executable not found")
        exit(1)
    lex_test, lex_stdout, lex_stderr = get_test_suite(Path("test/lexer"))
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
    
    print("=" * 60)
    print(f"{passed}/{len(lex_test)} passed")
    print("=" * 60)