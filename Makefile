# HypergraphGRAND Makefile
# Essential commands for environment setup and running experiments

VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python
MAIN_FILE = main.py
REQUIREMENTS_FILE = requirements.txt

# Color codes for output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

.PHONY: default help venv install run mlflow clean

default: help

help:
	@echo "$(BLUE)HypergraphGRAND Project Makefile$(NC)"
	@echo "============================================="
	@echo ""
	@echo "$(GREEN) Setup Commands:$(NC)"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo ""
	@echo "$(GREEN) Run Commands:$(NC)"
	@echo "  make run         - Run main.py with default parameters"
	@echo "  make run ARGS=\"...\" - Run main.py with custom arguments"
	@echo "                      Example: make run ARGS=\"--dataset planetoid_cora --strategy classification\""
	@echo ""
	@echo "$(GREEN) MLflow Commands:$(NC)"
	@echo "  make mlflow      - Start MLflow UI server (http://localhost:5000)"
	@echo ""
	@echo "$(GREEN) Maintenance Commands:$(NC)"
	@echo "  make clean       - Clean generated files and cache"

venv:
	@echo "$(BLUE) Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment already exists$(NC)"; \
	fi

install: venv 
	@echo "$(BLUE) Installing Python dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS_FILE)" ]; then \
		$(PIP) install -r $(REQUIREMENTS_FILE); \
		echo "$(GREEN)✓ Dependencies installed from requirements.txt$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  requirements.txt not found$(NC)"; \
	fi

run: venv install
	@echo "$(BLUE) Running HypergraphGRAND...$(NC)"
	@if [ -z "$(ARGS)" ]; then \
		$(PYTHON_VENV) $(MAIN_FILE) --help; \
		echo ""; \
		echo "$(YELLOW)Run with custom arguments: make run ARGS=\"--dataset planetoid_cora --strategy classification\"$(NC)"; \
	else \
		$(PYTHON_VENV) $(MAIN_FILE) $(ARGS); \
	fi

mlflow: venv install
	@echo "$(BLUE) Starting MLflow UI server...$(NC)"
	@echo "$(YELLOW) MLflow UI available at: http://localhost:5000$(NC)"
	@echo "$(YELLOW) Press Ctrl+C to stop the server$(NC)"
	@$(VENV_NAME)/bin/mlflow ui

clean:
	@echo "$(BLUE) Cleaning generated files and cache...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.png" -type f -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned generated files and cache$(NC)"
