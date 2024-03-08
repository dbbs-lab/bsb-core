from exceptiongroup import ExceptionGroup

from .exceptions import PackageRequirementWarning
from .reporting import warn


class MissingRequirementErrors(ExceptionGroup):
    pass


def get_missing_requirement_reason(package):
    from importlib.metadata import PackageNotFoundError, version

    from packaging.requirements import InvalidRequirement, Requirement

    try:
        req = Requirement(package)
    except InvalidRequirement:
        return f"Can't check package requirement '{package}': invalid requirement"
    try:
        ver = version(req.name)
    except PackageNotFoundError:
        return f"Missing package '{req.name}'. You may experience errors or differences in results."
    else:
        if not ver in req.specifier:
            return (
                f"Installed version of '{req.name}' ({ver}) "
                f"does not match requirements: '{req}'. You may experience errors or differences in results."
            )


def get_missing_packages(packages):
    return [
        package
        for package in packages
        if get_missing_requirement_reason(package) is not None
    ]


def get_unmet_package_reasons(packages):
    return [
        reason
        for package in packages
        if (reason := get_missing_requirement_reason(package)) is not None
    ]


def warn_missing_packages(packages):
    for warning in get_unmet_package_reasons(packages):
        warn(warning, PackageRequirementWarning)


def raise_missing_packages(packages):
    raise MissingRequirementErrors(
        "Your model is missing requirement(s)",
        [
            PackageRequirementWarning(warning)
            for warning in get_unmet_package_reasons(packages)
        ],
    )
