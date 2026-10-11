"""Every executable example in README.md runs, and every report line it quotes is real output.

The Python blocks run in order in one namespace, like a reader would run them. A block preceded
by ``<!-- readme-test: skip -->`` is illustrative (placeholders, ``my_project`` imports, ...).
Blocks that need an optional framework (torchvision, transformers, lightning) are skipped when it
is not installed; ``pip install -e ".[integrations]"`` runs them too.
"""

import contextlib
import io
import pathlib
import re
import textwrap

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"
OPTIONAL_FRAMEWORKS = ("torchvision", "transformers", "lightning")
SKIP_MARKER = "<!-- readme-test: skip -->"
BLOCK = re.compile(
    rf"(?P<marker>{re.escape(SKIP_MARKER)}\n)?```(?P<lang>\w+)\n(?P<code>.*?)```", re.S
)
REPORT_LINE = re.compile(r"^(PASS|WARN|FAIL|SKIP)  \w")


def _expected_lines(text_block: str) -> list[str]:
    """Lines of a text block that must appear in the output of the preceding Python block."""
    lines = [line.strip() for line in text_block.splitlines() if line.strip()]
    if lines and lines[0] == "SUMMARY":  # a TestSuite summary: every line counts
        return lines
    return [line for line in lines if REPORT_LINE.match(line)]


@pytest.mark.filterwarnings("ignore:Error detected in")  # detect_anomaly example prints a trace
def test_readme_examples():
    blocks = list(BLOCK.finditer(README.read_text(encoding="utf-8")))
    namespace: dict = {"__name__": "readme_example"}
    problems, missing_frameworks = [], set()
    executed = 0
    for index, block in enumerate(blocks):
        if block["lang"] != "python" or block["marker"]:
            continue
        code, output = textwrap.dedent(block["code"]), io.StringIO()  # list items are indented
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(code, f"README.md python block {index}", "exec"), namespace)
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] in OPTIONAL_FRAMEWORKS:
                missing_frameworks.add(e.name.split(".")[0])
                continue
            raise
        executed += 1
        printed = output.getvalue()
        if "report = " in code and "report" in namespace:
            printed += "\n" + str(namespace["report"])
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if following is not None and following["lang"] == "text":
            for line in _expected_lines(following["code"]):
                if line not in printed:
                    problems.append(f"block {index}: expected output line not produced:\n  {line}")
    assert executed > 20, "the README parser found suspiciously few examples"
    assert not problems, "\n".join(problems)
    if missing_frameworks:
        missing = ", ".join(sorted(missing_frameworks))
        pytest.skip(f"all other README examples pass; not installed: {missing}")
