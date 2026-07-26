from pathlib import Path

remove_count = 0
for i in Path("test").rglob("*.stdout"):
    if len(i.read_text()) == 0:
        print("remove empty stdout file:", i)
        remove_count += 1
        i.unlink()
for i in Path("test").rglob("*.stderr"):
    if len(i.read_text()) == 0:
        print("remove empty stderr file:", i)
        remove_count += 1
        i.unlink()

if remove_count > 0:
    print("removed", remove_count, "empty files")
else:
    print("no empty files found")