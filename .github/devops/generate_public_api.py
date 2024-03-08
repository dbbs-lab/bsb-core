import difflib
import sys
from pathlib import Path

from bsb import _get_public_api_map


def public_annotations():
    annotations = []
    for api, module in _get_public_api_map().items():
        annotation = f'"bsb.{module}.{api}"'
        if api[0].isupper():
            annotation = f"typing.Type[{annotation}]"
        annotations.append(f"{api}: {annotation}")

    lines = [
        "if typing.TYPE_CHECKING:",
        *sorted(f"  import bsb.{module}" for module in {*_get_public_api_map().values()}),
        "",
        *sorted(annotations),
        "",
    ]

    return lines


if __name__ == "__main__":
    import bsb

    path = Path(bsb.__path__[0]) / "__init__.py"
    text = path.read_text()
    find = (
        "# Do not modify: autogenerated public API type annotations of the `bsb` module\n"
        "# fmt: off\n"
        "# isort: off\n"
    )
    idx = text.find(find)
    annotation_lines = public_annotations()
    if idx == -1:
        print("__init__.py file is missing the replacement tag", file=sys.stderr)
        exit(1)
    if "--check" in sys.argv:
        diff = "\n".join(
            l
            for l in difflib.ndiff(
                text[idx + len(find) :].split("\n"),
                annotation_lines,
            )
            if l[0] != " "
        )
        print(diff, file=sys.stderr, end="")
        exit(bool(diff))
    else:
        text = text[:idx] + find + "\n".join(annotation_lines)
        path.write_text(text)
