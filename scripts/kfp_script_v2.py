# from kfp import compiler, dsl
from pathlib import Path
import os
import re
from nbconvert import PythonExporter
import nbformat
from rich import print as pprint

WORKSPACE = Path.cwd().parent / "examples"
USE_CASE = "knowledge-tuning"


def read_step(step_name: str):
    notebook_path = (
        WORKSPACE / USE_CASE / step_name / f"{step_name.split('_', 1)[1]}.ipynb"
    )
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb)

    # Temporary fix for doing inline installs in juputer notebooks
    source = "import os\n" + source
    source = re.sub(r"get_ipython\(\)\.system\((.*)\)", r"os.system(f\1)", source)

    return source


def read_step_utils(step_name: str, source: str):
    step_path = WORKSPACE / USE_CASE / step_name / "utils"
    if step_path.exists():
        print(f"Utility files found in {step_path}, adding to source...")
        utils = ""

        utils += "# Utils function from files are added\n"
        for util_file in step_path.glob("*.py"):
            utils += f"# From file: {util_file.name}\n\n"
            with open(util_file, "r", encoding="utf-8") as f:
                utils += f.read() + "\n\n"

        utils += "# End of utils from files\n"

        pattern = r"""
            (?:                                  # start of line
                from\s+utils[^\n]*import         # from utils... import
                (?:\s*\([^\)]*\))?               # optional multi-line parentheses
                |                                 # OR
                import\s+utils[^\n]*             # import utils...
            )
        """

        # Replace matched imports by commenting out each line
        source = re.sub(
            pattern,
            lambda m: "\n".join("# " + line for line in m.group(0).splitlines()),
            source,
            flags=re.MULTILINE | re.VERBOSE | re.DOTALL,
        )

        source = utils + "\n\n" + source
    return source


def execute_step(step_name: str, source: str):
    old_cwd = os.getcwd()
    os.chdir(WORKSPACE / USE_CASE / step_name)
    try:
        namespace_locals = {}
        exec(source, namespace_locals, namespace_locals)
    finally:
        os.chdir(old_cwd)


def run_step(step_name: str):
    pprint(f"\n\n=== Running Step: [green]{step_name}[/green] ===\n\n")

    source = read_step(step_name)
    source = read_step_utils(step_name, source)
    with open(f"{step_name.split('_', 1)[1]}.py", "w") as f:
        f.write(source)

    pprint(f"Executing Step: [red]{step_name}[/red]\n")
    execute_step(step_name, source)
    pprint(f"Completed Step: [red]{step_name}[/red]\n")
    pprint(f"\n\n=== Completed Step: [green]{step_name}[/green] ===\n\n")

    return source


def set_env_variables():
    # Required in Steps 01, 05, 06
    os.environ["STUDENT_MODEL_NAME"] = "meta-llama/Llama-3.2-1B-Instruct"
    # Knowledge Generation
    # Required in Steps 03
    os.environ["TEACHER_MODEL_NAME"] = "openai/gpt-oss-120b"
    os.environ["TEACHER_MODEL_BASE_URL"] = "http://0.0.0.0:8000/v1"
    os.environ["TEACHER_MODEL_API_KEY"] = ""

    # Knowledge Mixing
    # Required in Steps 04
    os.environ["TOKENIZER_MODEL_NAME"] = "meta-llama/Llama-3.2-1B-Instruct"
    os.environ["SAVE_GPT_OSS_FORMAT"] = "false"
    os.environ["CUT_SIZES"] = "5,50"
    os.environ["QA_PER_DOC"] = "10"

    os.environ["HF_TOKEN"] = ""


if __name__ == "__main__":
    step_pattern = re.compile(r"^\d{2}_[a-zA-Z0-9_-]+$")

    steps = [
        item
        for item in os.listdir(WORKSPACE / USE_CASE)
        if Path(WORKSPACE / USE_CASE / item).is_dir()
        and step_pattern.match(item)
        and item.split("_")[0] != "00"
    ]

    steps = sorted(steps, key=lambda x: int(x.split("_")[0]))
    pprint("Found steps:")
    for step in steps:
        pprint(f"-\t{step}")

    set_env_variables()
    steps = steps[:]

    for step in steps:
        run_step(step)
