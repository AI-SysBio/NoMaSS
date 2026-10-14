# Releasing NoMaSS on PyPI

One-time setup

1. Create accounts on https://test.pypi.org and https://pypi.org (the same username can be used on both).
2. On each site, create an API token (Account settings → API tokens). The first upload of a new project needs an account-wide token; afterwards you can scope the token to the `nomass` project.
3. Install the tools: `pip install --upgrade build twine`.

For every release

1. Bump the version in `pyproject.toml`, `nomass/__init__.py` and `CITATION.cff`, add an entry to `CHANGELOG.md`, and commit.
2. Run the tests: `pytest -q`.
3. Build the distributions (this creates `dist/nomass-<version>.tar.gz` and `dist/nomass-<version>-py3-none-any.whl`):

       python -m build

4. Check the metadata and the rendered README:

       python -m twine check dist/*

5. Upload to TestPyPI first and install from there in a fresh environment:

       python -m twine upload --repository testpypi dist/*
       pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nomass

   When prompted, use `__token__` as the username and the API token as the password.
6. Upload to PyPI:

       python -m twine upload dist/*

7. Tag the release on GitHub and publish a release note from `CHANGELOG.md`:

       git tag v<version>
       git push origin main --tags

A version can never be re-uploaded to PyPI: if something is wrong after the upload, bump the version (for example `1.0.1`) and upload again.
