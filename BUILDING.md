# Building & Publishing Harmony Proxy Packages

This guide explains how to generate source distributions and wheels for multiple
Python versions and upload them to an Artifactory-hosted PyPI repository.

## Prerequisites

1. Install the interpreter versions you plan to ship (for example 3.9–3.12).
   Tools such as `pyenv`, `asdf`, or system package managers can provide these.
2. Ensure each interpreter has `pip` and `venv`.
3. Install the build tooling once (outside the project virtual environments):
   ```bash
   python3 -m pip install --upgrade build twine
   ```
4. Obtain Artifactory credentials or an API key with publish permissions.

## Configure Artifactory Upload Target

Create or update `~/.pypirc` so `twine` knows about your Artifactory repository:

```
[distutils]
index-servers =
    harmony-artifactory

[harmony-artifactory]
repository = https://artifactory.example.com/artifactory/api/pypi/harmony-pypi
username = <ARTIFACTORY_USERNAME>
password = <ARTIFACTORY_API_KEY>
```

Use an environment variable (for example `ARTIFACTORY_API_KEY`) instead of a
plain-text password when possible:

```bash
export ARTIFACTORY_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
```

## Build Wheels for Each Python Version

Repeat the following steps for every supported interpreter (3.9, 3.10, 3.11,
3.12, …):

1. Create and activate a clean virtual environment with the target interpreter:
   ```bash
   /path/to/python3.9 -m venv .venv-py39
   source .venv-py39/bin/activate
   ```
2. Install project requirements in editable mode along with build tools:
   ```bash
   pip install --upgrade pip
   pip install -e .[harmony,dev]
   ```
3. Generate the artifacts (wheel + sdist) using `python -m build`:
   ```bash
   python -m build --wheel --sdist --outdir dist/py39
   ```
4. Deactivate the environment before building for the next interpreter:
   ```bash
   deactivate
   ```

After iterating through all interpreters, you will have subdirectories in `dist/`
containing wheels tagged for each Python version plus a single source tarball.

## Consolidate Artifacts

Combine all wheels and the source distribution into a single directory prior to
publishing:

```bash
mkdir -p dist/upload
find dist -mindepth 1 -type f -name "*.whl" -exec cp {} dist/upload/ \;
find dist -mindepth 1 -type f -name "*.tar.gz" -exec cp {} dist/upload/ \;
```

## Upload to Artifactory PyPI

Run `twine` against the consolidated directory:

```bash
twine upload --repository harmony-artifactory dist/upload/*
```

If you prefer environment variables over `~/.pypirc`, supply them explicitly:

```bash
twine upload \
  --repository-url https://artifactory.example.com/artifactory/api/pypi/harmony-pypi \
  --username "$ARTIFACTORY_USERNAME" \
  --password "$ARTIFACTORY_API_KEY" \
  dist/upload/*
```

## Verification

1. Visit the Artifactory UI and confirm the versioned files appear under the
   expected repository.
2. Optionally install from Artifactory into a fresh virtual environment:
   ```bash
   python -m venv verify-env
   source verify-env/bin/activate
   pip install --index-url https://artifactory.example.com/artifactory/api/pypi/harmony-pypi/simple \
       --extra-index-url https://pypi.org/simple \
       harmony-proxy==<version>
   deactivate
   ```
3. After verification, remove temporary build directories if they are no longer
   needed:
   ```bash
   rm -rf dist/upload verify-env
   ```

Following these steps ensures you build wheels with the correct Python tags and
publish them safely to your Artifactory PyPI instance.
