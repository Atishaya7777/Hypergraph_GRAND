VENV_NAME = venv
PYTHON = python3
PIP = uv pip
PYTHON_VENV = $(VENV_NAME)/bin/python
MAIN_FILE = main.py
REQUIREMENTS_FILE = requirements.txt

# Default parameters
EPOCHS ?= 200
PATIENCE ?= 50
DIMENSION ?= all
SEEDS ?= 5

BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m 

.PHONY: default help venv install test train train-representative train-all diffusion-study diffusion-fast analyze mlflow clean

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
	@echo "$(GREEN)Diffusion Study Commands:$(NC)"
	@echo "  make diffusion-fast    - Quick hypothesis validation (3 datasets, ~4 hours)"
	@echo "  make diffusion-study   - Full diffusion parameter studies (all datasets)"
	@echo "                          Usage: make diffusion-study [DIMENSION=all] [SEEDS=5]"
	@echo "  make analyze           - Generate reports and visualizations from MLflow"
	@echo ""
	@echo "$(GREEN)MLflow Commands:$(NC)"
	@echo "  make mlflow            - Start MLflow UI server (http://localhost:5000)"
	@echo ""
	@echo "$(GREEN)Maintenance Commands:$(NC)"
	@echo "  make clean             - Clean generated files and cache"

venv:
	@echo "$(BLUE) Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		uv venv $(VENV_NAME); \
		echo "$(GREEN)  Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)   Virtual environment already exists$(NC)"; \
	fi

install: venv 
	@echo "$(BLUE) Installing Python dependencies...$(NC)"
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

diffusion-fast: venv install
	@echo "$(BLUE) Running fast diffusion study (representative datasets only)...$(NC)"
	@echo "$(YELLOW) This tests 3 datasets with reduced configs (~50 experiments, ~4 hours)$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode diffusion-study --fast-mode --representative-only --num-seeds 3 --output diffusion_fast_results.json
	@echo "$(GREEN)  Fast study complete. Results saved to diffusion_fast_results.json$(NC)"

diffusion-study: venv install
	@echo "$(BLUE) Running full diffusion parameter studies...$(NC)"
	@echo "$(YELLOW) This may take 20-40 hours depending on hardware$(NC)"
	@$(PYTHON_VENV) $(MAIN_FILE) --mode diffusion-study --study-dimension $(DIMENSION) --num-seeds $(SEEDS) --output diffusion_study_results.json
	@echo "$(GREEN)  Diffusion study complete. Results saved to diffusion_study_results.json$(NC)"

analyze: venv install
	@echo "$(BLUE) Generating analysis reports and visualizations...$(NC)"
	@mkdir -p analysis_output
	@$(PYTHON_VENV) experiments/analyze_diffusion_results.py --generate-report --generate-latex --plot-heatmap --output-dir analysis_output
	@echo "$(GREEN)  Analysis complete. See analysis_output/ directory$(NC)"

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
	@rm -rf analysis_output 2>/dev/null || true
	@echo "$(GREEN)  Cleaned generated files and cache$(NC)"
