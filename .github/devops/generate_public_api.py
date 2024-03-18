import ast
import difflib
import functools
import sys
from pathlib import Path


def _assign_targets(assign: ast.Assign, id_: str):
    return any(
        target.id == id_ for target in assign.targets if isinstance(target, ast.Name)
    )


@functools.cache
def get_public_api_map():
    root = Path(__file__).parent

    public_api_map = {}
    for file in root.rglob("*.py"):
        module_parts = file.relative_to(root).parts
        module = ".".join(
            module_parts[:-1]
            + ((module_parts[-1][:-3],) if module_parts[-1] != "__init__.py" else tuple())
        )
        module_api = []
        for assign in ast.parse(file.read_text()).body:
            if isinstance(assign, ast.Assign):
                is_api = _assign_targets(assign, "__api__")
                is_either = is_api or _assign_targets(assign, "__all__")
                if ((is_either and not module_api) or is_api) and isinstance(
                    assign.value, ast.List
                ):
                    module_api = [
                        el.value
                        for el in assign.value.elts
                        if isinstance(el, ast.Constant)
                    ]
        for api in module_api:
            public_api_map[api] = module

    return public_api_map


def public_annotations():
    annotations = []
    for api, module in get_public_api_map().items():
        annotation = f'"bsb.{module}.{api}"'
        if api[0].isupper():
            annotation = f"typing.Type[{annotation}]"
        annotations.append(f"{api}: {annotation}")

    lines = [
        "if typing.TYPE_CHECKING:",
        *sorted(f"  import bsb.{module}" for module in {*get_public_api_map().values()}),
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
