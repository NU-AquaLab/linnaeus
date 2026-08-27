# Contributing to Linneaus

Thank you for your interest in contributing to Linneaus! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Issue Reporting](#issue-reporting)

## Getting Started

### Prerequisites

- Python 3.9 or higher
- [UV package manager](https://github.com/astral-sh/uv)
- Git

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/linneaus-project/linneaus.git
   cd linneaus
   ```

2. **Set up development environment**:
   ```bash
   # Install UV if you haven't already
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in development mode
   make install-dev
   # or: uv pip install -e ".[dev,docs]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify setup**:
   ```bash
   linneaus --version
   make test
   ```

## Code Style

We use several tools to maintain code quality:

### Formatting
- **Black**: Code formatting
- **isort**: Import sorting

Run formatting:
```bash
make format
```

### Linting
- **flake8**: General linting
- **pydocstyle**: Docstring style checking (NumPy convention)
- **mypy**: Type checking
- **bandit**: Security linting

Run all checks:
```bash
make lint
make type-check
make security
```

### Pre-commit Hooks

All code quality checks run automatically on commit. To run manually:
```bash
pre-commit run --all-files
```

## Testing

We use pytest for testing with high coverage requirements.

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test files
pytest tests/test_models/ -v

# Run specific test
pytest tests/test_models/test_schemas.py::TestASClassification::test_valid_classification -v
```

### Test Structure

```
tests/
├── test_config/     # Configuration tests
├── test_data/       # Data processing tests
├── test_models/     # Model and ML tests
├── test_cli/        # CLI interface tests
└── conftest.py      # Shared fixtures
```

### Writing Tests

- Use descriptive test names
- Include docstrings for test classes and complex tests
- Use fixtures from `conftest.py` when possible
- Mock external API calls
- Test both success and failure cases

Example:
```python
class TestASClassification:
    """Test AS classification functionality."""

    def test_valid_classification(self, sample_organization_data):
        """Test creating valid classification."""
        # Test implementation
        pass

    def test_invalid_asn_validation(self):
        """Test validation of invalid ASN numbers."""
        # Test implementation
        pass
```

## Documentation

### Docstrings

Use NumPy-style docstrings:

```python
def classify_organizations(
    self,
    organizations: List[OrganizationData],
    model_id: Optional[str] = None
) -> BatchClassificationResponse:
    """
    Classify a batch of AS organizations.

    Parameters
    ----------
    organizations : List[OrganizationData]
        Organizations to classify.
    model_id : str, optional
        Model identifier to use. Uses default if None.

    Returns
    -------
    BatchClassificationResponse
        Classification results for all organizations.

    Raises
    ------
    ValueError
        If organizations list is empty.
    APIError
        If OpenAI API request fails.

    Examples
    --------
    >>> classifier = ASClassifier()
    >>> orgs = [OrganizationData(asn=174, name="Test")]
    >>> results = classifier.classify_organizations(orgs)
    """
```

### README Updates

When adding features, update relevant sections in README.md:
- Installation instructions
- Usage examples
- Feature descriptions
- Configuration options

## Submitting Changes

### Pull Request Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   make ci-check  # Runs all quality checks
   ```

4. **Commit with clear messages**:
   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Messages

Use clear, descriptive commit messages:
- **feat**: New features
- **fix**: Bug fixes
- **docs**: Documentation changes
- **test**: Test additions/modifications
- **refactor**: Code refactoring
- **style**: Code style changes
- **ci**: CI/CD changes

Examples:
```
feat: add support for IPv6 ASN classification
fix: handle missing organization names gracefully
docs: update installation instructions for UV
test: add integration tests for CLI commands
```

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage maintained (>90%)
- [ ] Code style checks pass
- [ ] Documentation updated
- [ ] Changelog entry added (for significant changes)
- [ ] Review requested

## Issue Reporting

### Bug Reports

Use the bug report template and include:
- Python version
- Linneaus version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal code example

### Feature Requests

Use the feature request template and include:
- Clear description of the problem
- Proposed solution
- Alternative solutions considered
- Additional context/examples

### Enhancement Ideas

For improvements to existing features:
- Current behavior description
- Proposed enhancement
- Benefits and use cases
- Backward compatibility considerations

## Development Guidelines

### Code Organization

- Keep modules focused and cohesive
- Use dependency injection for testability
- Follow SOLID principles
- Prefer composition over inheritance
- Use type hints throughout

### Performance Considerations

- Profile performance-critical paths
- Use async/await for I/O operations
- Implement proper caching strategies
- Consider memory usage for large datasets

### Error Handling

- Use specific exception types
- Provide helpful error messages
- Log errors appropriately
- Handle edge cases gracefully

### API Design

- Follow RESTful principles where applicable
- Use consistent naming conventions
- Provide comprehensive validation
- Support both sync and async interfaces

## Getting Help

- **Documentation**: Check README and docstrings first
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Create issues for bugs and feature requests
- **Discord**: Join our development Discord server (link in README)

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Package metadata

Thank you for contributing to Linneaus! 🎉
