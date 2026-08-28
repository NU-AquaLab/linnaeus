# Makefile for Linnaeus Development
# ========================================
# Comprehensive development toolkit for the Linnaeus autonomous systems classification project

# Configuration
PACKAGE_NAME := linnaeus
SRC_DIR := src
TESTS_DIR := tests
DOCS_DIR := docs
BUILD_DIR := build
DIST_DIR := dist

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Python and UV detection
PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null)
UV := $(shell command -v uv 2>/dev/null)

# Check if UV is available
ifndef UV
$(error "UV package manager not found. Please install UV: https://github.com/astral-sh/uv")
endif

# Main targets
.PHONY: help install install-dev clean test test-cov test-parallel lint format type-check security autoflake pydocstyle \
        docs docs-serve docs-clean build publish docker docker-dev docker-stop run-dev pre-commit ci-check \
        prepare-release version-bump changelog profile benchmark complexity outdated validate-config \
        data-download data-validate cli-test linnaeus-help clean-all

# =============================================================================
# HELP & INFO
# =============================================================================

## Show this help message
help:
	@echo ""
	@echo "$(BLUE)Linnaeus Development Makefile$(RESET)"
	@echo "============================="
	@echo ""
	@echo "$(GREEN)Installation:$(RESET)"
	@echo "  install         Install package in production mode"
	@echo "  install-dev     Install package in development mode with all dependencies"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@echo "  clean           Clean build artifacts and cache files"
	@echo "  clean-all       Deep clean including UV cache and environments"
	@echo "  format          Format code with black, isort, and autoflake"
	@echo "  lint            Run all linting checks (flake8, pydocstyle)"
	@echo "  type-check      Run type checking with mypy"
	@echo "  security        Run security checks with bandit"
	@echo "  pre-commit      Run pre-commit hooks on all files"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  test            Run tests with pytest"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  test-parallel   Run tests in parallel for faster execution"
	@echo "  cli-test        Test CLI commands and functionality"
	@echo ""
	@echo "$(GREEN)Quality Assurance:$(RESET)"
	@echo "  ci-check        Run full CI pipeline locally"
	@echo "  complexity      Analyze code complexity"
	@echo "  profile         Run performance profiling"
	@echo "  benchmark       Run benchmark tests"
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@echo "  docs            Generate documentation with Sphinx"
	@echo "  docs-serve      Serve documentation locally"
	@echo "  docs-clean      Clean documentation build"
	@echo ""
	@echo "$(GREEN)Building & Release:$(RESET)"
	@echo "  build           Build package distributions"
	@echo "  publish         Publish package to PyPI (requires authentication)"
	@echo "  version-bump    Interactive version bumping"
	@echo "  prepare-release Prepare release checklist"
	@echo "  changelog       Generate changelog from git commits"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  docker          Build Docker image"
	@echo "  docker-dev      Start development environment with Docker Compose"
	@echo "  docker-stop     Stop Docker development environment"
	@echo "  run-dev         Start complete development environment"
	@echo ""
	@echo "$(GREEN)Linnaeus Specific:$(RESET)"
	@echo "  data-download   Download all data sources"
	@echo "  data-validate   Validate downloaded data quality"
	@echo "  linnaeus-help   Show Linnaeus CLI help"
	@echo "  validate-config Validate configuration files"
	@echo ""
	@echo "$(GREEN)Environment:$(RESET)"
	@echo "  outdated        Check for outdated dependencies"
	@echo "  env-info        Show environment information"
	@echo ""

## Show environment information
env-info:
	@echo "$(BLUE)Environment Information:$(RESET)"
	@echo "Python: $(PYTHON)"
	@echo "UV: $(UV)"
	@$(UV) --version
	@echo "Package: $(PACKAGE_NAME)"
	@echo "Source: $(SRC_DIR)"
	@echo "Tests: $(TESTS_DIR)"

# =============================================================================
# INSTALLATION
# =============================================================================

## Install package in production mode
install:
	@echo "$(BLUE)Installing $(PACKAGE_NAME) in production mode...$(RESET)"
	$(UV) pip install .

## Install package in development mode with all dependencies
install-dev:
	@echo "$(BLUE)Installing $(PACKAGE_NAME) in development mode...$(RESET)"
	$(UV) pip install -e ".[dev,docs]"
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	pre-commit install
	@echo "$(GREEN)✓ Development environment ready!$(RESET)"

# =============================================================================
# CLEANING
# =============================================================================

## Clean build artifacts and cache files
clean:
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf $(BUILD_DIR)/
	rm -rf $(DIST_DIR)/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)✓ Cleaned build artifacts$(RESET)"

## Deep clean including UV cache and environments
clean-all: clean
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	$(UV) cache clean
	rm -rf .tox/
	rm -rf .nox/
	rm -rf docs/_build/
	@echo "$(GREEN)✓ Deep clean completed$(RESET)"

# =============================================================================
# CODE FORMATTING & LINTING
# =============================================================================

## Format code with black, isort, and autoflake
format:
	@echo "$(BLUE)Formatting code...$(RESET)"
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --ignore-init-module-imports -r $(SRC_DIR)/ $(TESTS_DIR)/
	black $(SRC_DIR)/ $(TESTS_DIR)/
	isort $(SRC_DIR)/ $(TESTS_DIR)/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

## Remove unused imports with autoflake
autoflake:
	@echo "$(BLUE)Removing unused imports...$(RESET)"
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --ignore-init-module-imports -r $(SRC_DIR)/ $(TESTS_DIR)/
	@echo "$(GREEN)✓ Unused imports removed$(RESET)"

## Run all linting checks
lint:
	@echo "$(BLUE)Running linting checks...$(RESET)"
	flake8 $(SRC_DIR)/ $(TESTS_DIR)/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)✓ Flake8 passed$(RESET)"

## Check documentation style
pydocstyle:
	@echo "$(BLUE)Checking documentation style...$(RESET)"
	pydocstyle $(SRC_DIR)/ --convention=numpy --add-ignore=D100,D104,D105
	@echo "$(GREEN)✓ Documentation style passed$(RESET)"

## Run type checking with mypy
type-check:
	@echo "$(BLUE)Running type checks...$(RESET)"
	mypy $(SRC_DIR)/ --ignore-missing-imports --no-strict-optional
	@echo "$(GREEN)✓ Type checking passed$(RESET)"

## Run security checks with bandit
security:
	@echo "$(BLUE)Running security checks...$(RESET)"
	bandit -r $(SRC_DIR)/ -ll
	@echo "$(GREEN)✓ Security checks passed$(RESET)"

## Check for dependency vulnerabilities
security-deps:
	@echo "$(BLUE)Checking dependency vulnerabilities...$(RESET)"
	safety check --json || echo "$(YELLOW)⚠ Some vulnerabilities found - review output$(RESET)"

## Run pre-commit hooks on all files
pre-commit:
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks passed$(RESET)"

# =============================================================================
# TESTING
# =============================================================================

## Run tests with pytest
test:
	@echo "$(BLUE)Running tests...$(RESET)"
	pytest $(TESTS_DIR)/ -v
	@echo "$(GREEN)✓ Tests passed$(RESET)"

## Run tests with coverage report
test-cov:
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	pytest $(TESTS_DIR)/ --cov=$(SRC_DIR)/$(PACKAGE_NAME) --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)✓ Tests with coverage completed$(RESET)"
	@echo "$(BLUE)Coverage report: htmlcov/index.html$(RESET)"

## Run tests in parallel for faster execution
test-parallel:
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	pytest $(TESTS_DIR)/ -n auto -v
	@echo "$(GREEN)✓ Parallel tests completed$(RESET)"

## Test CLI commands and functionality
cli-test:
	@echo "$(BLUE)Testing CLI functionality...$(RESET)"
	$(PACKAGE_NAME) --help
	$(PACKAGE_NAME) data --help
	@echo "$(GREEN)✓ CLI tests passed$(RESET)"

# =============================================================================
# QUALITY ASSURANCE
# =============================================================================

## Run full CI pipeline locally
ci-check: format lint pydocstyle type-check security test-cov
	@echo ""
	@echo "$(GREEN)🎉 All CI checks passed! Ready for commit.$(RESET)"

## Analyze code complexity
complexity:
	@echo "$(BLUE)Analyzing code complexity...$(RESET)"
	radon cc $(SRC_DIR)/ -a -nb
	radon mi $(SRC_DIR)/ -nb
	@echo "$(GREEN)✓ Complexity analysis completed$(RESET)"

## Run performance profiling
profile:
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -s cumulative -m pytest $(TESTS_DIR)/ -k "not slow" --tb=no -q

## Run benchmark tests
benchmark:
	@echo "$(BLUE)Running benchmark tests...$(RESET)"
	pytest $(TESTS_DIR)/ -m benchmark -v || echo "$(YELLOW)No benchmark tests found$(RESET)"

# =============================================================================
# DOCUMENTATION
# =============================================================================

## Generate documentation with Sphinx
docs:
	@echo "$(BLUE)Generating documentation...$(RESET)"
	@if [ ! -f "$(DOCS_DIR)/conf.py" ]; then \
		echo "$(YELLOW)Creating Sphinx documentation structure...$(RESET)"; \
		sphinx-quickstart -q -p "$(PACKAGE_NAME)" -a "Linnaeus Team" --ext-autodoc --ext-viewcode --makefile --no-batchfile $(DOCS_DIR); \
	fi
	sphinx-build -b html $(DOCS_DIR) $(DOCS_DIR)/_build/html
	@echo "$(GREEN)✓ Documentation generated$(RESET)"
	@echo "$(BLUE)Documentation available at: $(DOCS_DIR)/_build/html/index.html$(RESET)"

## Serve documentation locally
docs-serve: docs
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	@cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

## Clean documentation build
docs-clean:
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	rm -rf $(DOCS_DIR)/_build/
	@echo "$(GREEN)✓ Documentation cleaned$(RESET)"

# =============================================================================
# BUILDING & RELEASE
# =============================================================================

## Build package distributions
build: clean
	@echo "$(BLUE)Building package...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Package built successfully$(RESET)"
	@echo "$(BLUE)Built packages:$(RESET)"
	@ls -la $(DIST_DIR)/

## Publish package to PyPI
publish: build
	@echo "$(BLUE)Publishing to PyPI...$(RESET)"
	@echo "$(YELLOW)⚠ Make sure you have proper PyPI credentials configured$(RESET)"
	twine upload $(DIST_DIR)/*
	@echo "$(GREEN)✓ Package published$(RESET)"

## Interactive version bumping
version-bump:
	@echo "$(BLUE)Current version:$(RESET)"
	@grep "version" pyproject.toml
	@echo ""
	@echo "$(YELLOW)Please update version in pyproject.toml manually$(RESET)"
	@echo "$(YELLOW)Consider using semantic versioning: MAJOR.MINOR.PATCH$(RESET)"

## Generate changelog from git commits
changelog:
	@echo "$(BLUE)Generating changelog...$(RESET)"
	@git log --pretty=format:"- %s (%h)" --since="$(shell git describe --tags --abbrev=0 2>/dev/null || echo '1 month ago')" > CHANGELOG_TEMP.md || true
	@echo "$(GREEN)✓ Changelog generated in CHANGELOG_TEMP.md$(RESET)"

## Prepare release checklist
prepare-release:
	@echo ""
	@echo "$(BLUE)📋 Release Preparation Checklist:$(RESET)"
	@echo ""
	@echo "$(GREEN)1.$(RESET) Update version in pyproject.toml"
	@echo "$(GREEN)2.$(RESET) Update CHANGELOG.md with new features and fixes"
	@echo "$(GREEN)3.$(RESET) Run: make ci-check"
	@echo "$(GREEN)4.$(RESET) Commit changes: git commit -m 'Prepare release vX.Y.Z'"
	@echo "$(GREEN)5.$(RESET) Create and push tag: git tag vX.Y.Z && git push origin vX.Y.Z"
	@echo "$(GREEN)6.$(RESET) Build and publish: make publish"
	@echo "$(GREEN)7.$(RESET) Create GitHub release with changelog"
	@echo ""

# =============================================================================
# DOCKER
# =============================================================================

## Build Docker image
docker:
	@echo "$(BLUE)Building Docker image...$(RESET)"
	docker build -t $(PACKAGE_NAME):latest .
	@echo "$(GREEN)✓ Docker image built$(RESET)"

## Start development environment with Docker Compose
docker-dev:
	@echo "$(BLUE)Starting development environment...$(RESET)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)✓ Development environment started$(RESET)"

## Stop Docker development environment
docker-stop:
	@echo "$(BLUE)Stopping development environment...$(RESET)"
	docker-compose down
	@echo "$(GREEN)✓ Development environment stopped$(RESET)"

## Start complete development environment
run-dev: docker-dev
	@echo ""
	@echo "$(GREEN)🚀 Development environment ready!$(RESET)"
	@echo "$(BLUE)Jupyter Lab:$(RESET) http://localhost:8888"
	@echo "$(BLUE)Development container:$(RESET) docker exec -it linnaeus-dev bash"
	@echo ""

# =============================================================================
# AMEGHINO SPECIFIC COMMANDS
# =============================================================================

## Download all data sources
data-download:
	@echo "$(BLUE)Downloading data sources...$(RESET)"
	$(PACKAGE_NAME) data download --sources peeringdb asrank aspop ipinfo
	@echo "$(GREEN)✓ Data download completed$(RESET)"

## Validate downloaded data quality
data-validate:
	@echo "$(BLUE)Validating data quality...$(RESET)"
	$(PACKAGE_NAME) data validate --all
	@echo "$(GREEN)✓ Data validation completed$(RESET)"

## Show Linnaeus CLI help
linnaeus-help:
	@echo "$(BLUE)Linnaeus CLI Commands:$(RESET)"
	$(PACKAGE_NAME) --help

## Validate configuration files
validate-config:
	@echo "$(BLUE)Validating configuration files...$(RESET)"
	@$(PYTHON) -c "import yaml; yaml.safe_load(open('config.yaml'))" && echo "$(GREEN)✓ config.yaml is valid$(RESET)" || echo "$(RED)✗ config.yaml has errors$(RESET)"
	@$(UV) pip show tomli >/dev/null 2>&1 && $(PYTHON) -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))" && echo "$(GREEN)✓ pyproject.toml is valid$(RESET)" || echo "$(YELLOW)⚠ pyproject.toml validation skipped (install tomli for validation)$(RESET)"

# =============================================================================
# ENVIRONMENT MANAGEMENT
# =============================================================================

## Check for outdated dependencies
outdated:
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	$(UV) pip list --outdated || echo "$(GREEN)All dependencies are up to date$(RESET)"

# =============================================================================
# UTILITY TARGETS
# =============================================================================

# Ensure UV is available
check-uv:
	@$(UV) --version > /dev/null 2>&1 || (echo "$(RED)UV not found. Please install UV package manager$(RESET)" && exit 1)

# Default target
.DEFAULT_GOAL := help
