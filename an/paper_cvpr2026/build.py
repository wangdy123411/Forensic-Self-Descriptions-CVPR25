"""Compile the standalone CVPR excerpt with a local TeX Live/MiKTeX installation."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
from pypdf import PdfReader

BASE = Path(__file__).resolve().parent
BUILD = BASE / "build"
BUILD.mkdir(exist_ok=True)
env = os.environ.copy()
env["TEXINPUTS"] = str(BASE) + os.pathsep + str(BASE / "texmf/tex") + "//" + os.pathsep + env.get("TEXINPUTS", "")
env["BIBINPUTS"] = str(BASE) + os.pathsep + env.get("BIBINPUTS", "")
env["BSTINPUTS"] = str(BASE) + os.pathsep + env.get("BSTINPUTS", "")
commands = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"],
            ["bibtex", "build/main"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"],
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", "main.tex"]]
for i, command in enumerate(commands):
    p = subprocess.run(command, cwd=BASE, env=env, capture_output=True, text=True, errors="replace")
    (BUILD / f"compile_{i+1}.txt").write_text(p.stdout + p.stderr, encoding="utf-8")
    if p.returncode:
        print((p.stdout + p.stderr)[-6000:])
        sys.exit(p.returncode)
output = BASE / "output/pdf/robustness_subsection.pdf"
output.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(BUILD / "main.pdf", output)
pdf = PdfReader(output)
text = "\n".join(page.extract_text() for page in pdf.pages)
(BUILD / "extracted_text.txt").write_text(text, encoding="utf-8")
print(f"Compiled {output}; {len(pdf.pages)} page(s), {len(text.split())} extracted words.")
renderer = shutil.which("pdftoppm")
if not renderer:
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe"
    if bundled.exists():
        renderer = str(bundled)
if renderer:
    p = subprocess.run([renderer, "-png", "-r", "170", str(output), str(BUILD / "preview")], capture_output=True)
    if p.returncode:
        print(p.stderr.decode(errors="replace"))
        sys.exit(p.returncode)
    print("Rendered page previews under build/.")
