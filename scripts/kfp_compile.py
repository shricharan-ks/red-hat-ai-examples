# from kfp import compiler, dsl
from pathlib import Path
import os
import re
from nbconvert import PythonExporter
import nbformat

WORKSPACE = Path.cwd().parent / "examples"
USE_CASE = "knowledge-tuning"


step_pattern = re.compile(r"^\d{2}_[a-zA-Z0-9_-]+$")

steps = [
    item
    for item in os.listdir(WORKSPACE / USE_CASE)
    if Path(WORKSPACE / USE_CASE / item).is_dir()
    and step_pattern.match(item)
    and item.split("_")[0] != "00"
]


steps = sorted(steps, key=lambda x: int(x.split("_")[0]))
print("Found steps:")
for step in steps:
    print(f"- {step}")
step_1 = steps[0]

print(step_1)
notebook_path = WORKSPACE / USE_CASE / step_1 / f"{step_1.split('_', 1)[1]}.ipynb"
print(f"Converting notebook: {notebook_path}")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)


exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

# Temporary fix for doing inline installs in juputer notebooks
source = re.sub(r"get_ipython\(\)\.system\((.*)\)", r"import os\nos.system(\1)", source)


print("\n\n\n\n")
# print(_)

with open("output.py", "w") as f:
    f.write(source)


# Required in Steps 01, 05, 06
os.environ["STUDENT_MODEL_NAME"] = "RedHatAI/Llama-3.1-8B-Instruct"
# Knowledge Generation
# Required in Steps 03
os.environ["TEACHER_MODEL_NAME"] = "openai/gpt-oss-120b"
os.environ["TEACHER_MODEL_BASE_URL"] = "http://0.0.0.0:8000/v1"
os.environ["TEACHER_MODEL_API_KEY"] = None


# Knowledge Mixing
# Required in Steps 04
os.environ["TOKENIZER_MODEL_NAME"] = "RedHatAI/Llama-3.1-8B-Instruct"
os.environ["SAVE_GPT_OSS_FORMAT"] = "false"
os.environ["CUT_SIZES"] = "5,50"
os.environ["QA_PER_DOC"] = "10"

os.environ["HF_TOKEN"] = ""


old_cwd = os.getcwd()
os.chdir(WORKSPACE / USE_CASE / step_1)
try:
    namespace_locals = {}
    r = exec(source, namespace_locals, namespace_locals)
    print("Compiling pipeline...")
    print("\n\n\n", r)
finally:
    os.chdir(old_cwd)
