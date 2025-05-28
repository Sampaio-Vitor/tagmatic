# PyPI Release Guide for Tagmatic

This guide explains how to publish the Tagmatic package to PyPI using the `pypi-release` branch.

## Prerequisites

Before publishing to PyPI, ensure you have:

1. **PyPI Account**: Create accounts on both [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)
2. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
3. **Required Tools**: Install the necessary publishing tools:
   ```bash
   pip install build twine
   ```

## Setup API Tokens

Configure your API tokens for secure uploads:

### For TestPyPI:
```bash
# Create/edit ~/.pypirc
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_API_TOKEN_HERE
```

### For PyPI:
```bash
[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-YOUR_PYPI_API_TOKEN_HERE
```

## Release Process

### 1. Prepare the Release

1. Switch to the `pypi-release` branch:
   ```bash
   git checkout pypi-release
   ```

2. Update the version in `pyproject.toml`:
   ```toml
   [project]
   version = "0.1.0"  # Update this version
   ```

3. Ensure all tests pass:
   ```bash
   python -m pytest
   ```

4. Commit version changes:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.0"
   git push origin pypi-release
   ```

### 2. Test Release (Recommended)

First, test your package on TestPyPI:

```bash
python scripts/publish_to_pypi.py --test
```

This will:
- Clean previous build artifacts
- Build the package
- Upload to TestPyPI

Test the installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ tagmatic
```

### 3. Production Release

Once you've verified the test release works correctly:

```bash
python scripts/publish_to_pypi.py --prod
```

This will:
- Clean previous build artifacts
- Build the package
- Ask for confirmation
- Upload to PyPI

## Manual Release Process

If you prefer to do it manually:

### Build the Package
```bash
# Clean previous builds
rm -rf build dist *.egg-info

# Build the package
python -m build
```

### Upload to TestPyPI
```bash
python -m twine upload --repository testpypi dist/*
```

### Upload to PyPI
```bash
python -m twine upload dist/*
```

## Version Management

Follow semantic versioning (SemVer):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.1.1` (patch)
- `0.1.1` → `0.2.0` (minor)
- `0.2.0` → `1.0.0` (major)

## Post-Release Checklist

After a successful release:

1. **Create a Git Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Create GitHub Release**: Go to GitHub and create a release from the tag

3. **Update Documentation**: Ensure README and docs reflect the new version

4. **Merge to Main**: Create a PR to merge `pypi-release` back to `main`

5. **Announce**: Share the release on relevant channels

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify your API tokens are correct
2. **Version Already Exists**: You cannot overwrite existing versions on PyPI
3. **Build Failures**: Check that all dependencies are properly specified
4. **Upload Failures**: Ensure your package name is available on PyPI

### Useful Commands

```bash
# Check package metadata
python -m twine check dist/*

# List package contents
tar -tzf dist/tagmatic-*.tar.gz

# Validate pyproject.toml
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

## Security Notes

- Never commit API tokens to version control
- Use API tokens instead of passwords
- Regularly rotate your API tokens
- Consider using GitHub Actions for automated releases

## Resources

- [PyPI Documentation](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/tutorials/packaging-projects/)
