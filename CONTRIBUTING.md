# Contributing to Siamese Confidence

Thank you for your interest in contributing to the Siamese Confidence package! This document provides guidelines for development and contribution.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/zaterka/siamese-confidence.git
cd siamese-confidence
```

### 2. Install Dependencies with uv

We use [uv](https://docs.astral.sh/uv/) as our package manager for fast, reliable dependency management.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project with development dependencies
uv sync --dev
```

This installs the package in editable mode along with development tools:
- pytest (testing)
- pytest-cov (coverage)
- ruff (linting and formatting)
- mypy (type checking)
- pre-commit (git hooks)

## Development Workflow

### Code Style and Pre-commit Hooks

We use pre-commit hooks to automatically maintain code quality. The hooks run automatically on commit, but you can also run them manually:

```bash
# Run all pre-commit hooks on all files
uv run --with pre-commit pre-commit run --all-files

# Install pre-commit hooks (done automatically during setup)
uv run --with pre-commit pre-commit install

# Format and lint code with ruff
uv run ruff format src/ tests/ examples/
uv run ruff check src/ tests/ examples/ --fix

# Type checking
uv run mypy src/
```

Our pre-commit hooks include:
- **Ruff**: Fast linting and formatting
- **File formatting**: Trailing whitespace, end-of-file fixes
- **Configuration validation**: TOML and YAML syntax checking

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/siamese_confidence --cov-report=html

# Run specific test file
uv run pytest tests/test_core.py

# Run specific test
uv run pytest tests/test_core.py::TestSiameseModel::test_initialization
```

### Package Structure

- `src/siamese_confidence/`: Main package code
  - `__init__.py`: Public API exports
  - `core.py`: Main SiameseModel and training functions
  - `models.py`: Neural network components
  - `utils.py`: Data loading and preprocessing
- `tests/`: Test suite
- `examples/`: Usage examples
- `docs/`: Documentation (future)

## Contribution Guidelines

### 1. Code Quality

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Write comprehensive docstrings
- Maintain test coverage above 80%
- Keep functions focused and modular

### 2. Testing

- Write tests for all new functionality
- Include edge cases and error conditions
- Use descriptive test names
- Mock external dependencies

### 3. Documentation

- Update docstrings for any API changes
- Add examples for new features
- Update README.md if needed
- Include type information in docstrings

### 4. Pull Request Process

1. Create a feature branch from main
2. Make your changes
3. Run the full test suite
4. Update documentation as needed
5. Submit a pull request with:
   - Clear description of changes
   - Reference to any related issues
   - Test results

### 5. Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification (Commitizen standard). Use the following format:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

**Examples:**
```
feat(core): add support for custom activation functions
fix(models): handle edge case in confidence computation
docs(api): update SiameseModel documentation
test(utils): add tests for StandardScaler edge cases
chore(deps): update dependencies to latest versions
```

## Design Principles

### 1. Pure NumPy Implementation

- Maintain the pure NumPy approach (no sklearn dependencies)
- Keep the library lightweight and focused
- Ensure compatibility with different NumPy versions

### 2. API Consistency

- Follow consistent naming conventions
- Maintain backward compatibility when possible
- Use clear, descriptive parameter names

### 3. Performance

- Optimize for common use cases
- Avoid unnecessary computations
- Use vectorized operations where possible

### 4. Robustness

- Handle edge cases gracefully
- Provide clear error messages
- Validate input parameters

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if exists)
3. Run full test suite
4. Create release tag
5. Build and upload to PyPI

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain a welcoming environment

Thank you for contributing to Siamese Confidence!
