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

.PHONY: default help venv install test train train-representative train-all mlflow clean

default: help

help:
	@echo "$(BLUE)HyperGRAND Project Makefile$(NC)"
	@echo "=============================================="
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make venv              - Create virtual environment"
	@echo "  make install           - Install Python dependencies"
	@echo ""
	@echo "$(GREEN)Testing & Validation:$(NC)"
	@echo "  make test              - Validate dataset structure (no training)"
	@echo ""
	@echo "$(GREEN)Training Commands:$(NC)"
	@echo "  make train             - Train on single dataset"
	@echo "                          Usage: make train DATASET=cora"
	@echo "  make train-representative - Train on 3 representative datasets"
	@echo "                          (cora, contact_high_school, zoo)"
	@echo "  make train-all         - Train on all 16 datasets (with resume capability)"
	@echo "                          Usage: make train-all [EPOCHS=200] [PATIENCE=50]"
	@echo ""
	@echo "$(GREEN)MLflow Commands:$(NC)"
	@echo "  make mlflow            - Start MLflow UI server (http://localhost:5000)"
	@echo ""
	@echo "$(GREEN)Maintenance Commands:$(NC)"
	@echo "  make clean             - Clean generated files and cache"

venv:
	@echo "$(BLUE) Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)  Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)   Virtual environment already exists$(NC)"; \
	fi

install: venv 
	@echo "$(BLUE) Installing Python dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS_FILE)" ]; then \
		$(PIP) install -r $(REQUIREMENTS_FILE); \
		echo "$(GREEN)  Dependencies installed from requirements.txt$(NC)"; \
	else \
		echo "$(YELLOW)   requirements.txt not found$(NC)"; \
	fi

test: venv install
	@echo "$(BLUE) Running dataset structure validation...$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode test

train: venv install
	@if [ -z "$(DATASET)" ]; then \
		echo "$(RED)Error: DATASET not specified$(NC)"; \
		echo "Usage: make train DATASET=cora"; \
		exit 1; \
	fi
	@echo "$(BLUE) Training on dataset: $(DATASET)...$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode train --dataset $(DATASET) --epochs 200 --patience 50

train-representative: venv install
	@echo "$(BLUE) Training on representative datasets...$(NC)"
	@echo "$(YELLOW) This trains: cora (classification), contact_high_school (clustering), zoo (partitioning)$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode train --epochs 200 --patience 50

train-all: venv install
	@echo "$(BLUE) Training on ALL 16 datasets...$(NC)"
	@echo "$(YELLOW) This may take a while. Training will resume from saved progress if interrupted.$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode batch --epochs $(EPOCHS) --patience $(PATIENCE) --save-results training_results.json
	@echo "$(GREEN)  Training complete. Results saved to training_results.json$(NC)"

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
	@echo "$(GREEN)  Cleaned generated files and cache$(NC)"
