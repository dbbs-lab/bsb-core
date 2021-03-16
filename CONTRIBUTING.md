# Contributing to Transcriptase
We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Go ahead!

## Any contributions you make will be under the GPLv3 Software License
In short, when you submit code changes, your submissions are understood to be under the same [GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.en.html) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/dbbs-lab/bsb/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](); it's that easy!

## Write bug reports with detail, background, and sample code
[This is an example](http://stackoverflow.com/q/12488905/180626) of a bug report that's not a bad model. Here's [another example from Craig Hockenberry](http://www.openradar.me/11905408).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Use a Consistent Coding Style

* 4 spaces for indentation rather than tabs
* Follow [black](https://github.com/psf/black) formatting. The `dev` extras contain the
  dependencies you need so that you can use `pre-commit install` to have automatic pre-commit
  hooks that format your code for you:

```
  pip install bsb[dev]
  pre-commit install
```

## License
By contributing, you agree that your contributions will be licensed under its GPLv3 License.

## References
This document was adapted from [briandk's gist](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62)
