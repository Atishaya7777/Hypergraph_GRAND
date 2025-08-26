# HypergraphGRAND Makefile
# Manages virtual environment, dependencies, and experiments
# Uses enhanced train.py with MLflow integration

VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python
TRAIN_DATASET_DIR = datasets/contact-high-school
TEST_DATASET_DIR = datasets/contact-primary-school
LOG_DIR = logs
SAVED_MODELS_DIR = saved_models
REQUIREMENTS_FILE = requirements.txt
MAIN_FILE = testing.py

# Color codes for output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
CYAN = \033[0;36m
MAGENTA = \033[0;35m
NC = \033[0m # No Color

.PHONY: default
default: check-and-run

.PHONY: check-and-run
check-and-run:
	@echo "$(BLUE) Starting HypergraphGRAND workflow...$(NC)"
	@echo "$(CYAN) Training on contact-high-school, testing on contact-primary-school$(NC)"
	@echo "$(BLUE)Step 1: Checking virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Virtual environment not found. Creating...$(NC)"; \
		$(MAKE) venv; \
		$(MAKE) install; \
	else \
		echo "$(GREEN)✓ Virtual environment exists$(NC)"; \
	fi
	@echo "$(BLUE)Step 2: Running experiments with MLflow...$(NC)"
	@mkdir -p $(LOG_DIR) $(SAVED_MODELS_DIR)
	@$(PYTHON_VENV) $(MAIN_FILE)
	@echo "$(GREEN) Experiments completed successfully!$(NC)"
	@echo "$(CYAN) View results with: make mlflow$(NC)"

.PHONY: help
help:
	@echo "$(BLUE)HypergraphGRAND Project Makefile$(NC)"
	@echo "============================================="
	@echo ""
	@echo "$(GREEN) Quick Start:$(NC)"
	@echo "  make             - Complete workflow: setup → check datasets → run experiments"
	@echo "  make run         - Run experiments with MLflow tracking"
	@echo "  make mlflow      - Start MLflow UI to view experiment results"
	@echo ""
	@echo "$(GREEN) Setup Commands:$(NC)"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo "  make deps        - Install additional development dependencies"
	@echo ""
	@echo "$(GREEN) Dataset Commands:$(NC)"
	@echo "  make dataset-check - Verify both training and testing datasets"
	@echo "  make dataset-info  - Show dataset statistics and information"
	@echo ""
	@echo "$(GREEN) Experiment Commands:$(NC)"
	@echo "  make run         - Run enhanced training with MLflow logging"
	@echo "  make test        - Test model components and functionality"
	@echo "  make validate    - Validate project structure and dependencies"
	@echo ""
	@echo "$(GREEN) MLflow Commands:$(NC)"
	@echo "  make mlflow      - Start MLflow UI server (http://localhost:5000)"
	@echo "  make mlflow-logs - View MLflow experiment summaries"
	@echo "  make mlflow-clean - Clean MLflow runs and artifacts"
	@echo ""
	@echo "$(GREEN)🧹 Maintenance Commands:$(NC)"
	@echo "  make clean       - Clean generated files and cache"
	@echo "  make clean-all   - Clean everything including venv and MLflow"
	@echo "  make lint        - Run code linting (flake8)"
	@echo "  make format      - Format code with black"
	@echo ""
	@echo "$(GREEN) Utility Commands:$(NC)"
	@echo "  make logs        - View recent training logs"
	@echo "  make models      - List saved models"
	@echo "  make freeze      - Update requirements.txt"
	@echo "  make status      - Show project status"

.PHONY: venv
venv:
	@echo "$(BLUE) Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)✓ Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment already exists$(NC)"; \
	fi

.PHONY: install
install: venv 
	@echo "$(BLUE) Installing Python dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@if [ -f "$(REQUIREMENTS_FILE)" ]; then \
		$(PIP) install -r $(REQUIREMENTS_FILE); \
		echo "$(GREEN) Dependencies installed from requirements.txt$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  requirements.txt not found, installing common packages...$(NC)"; \
		$(PIP) install torch torchvision scikit-learn numpy matplotlib seaborn mlflow; \
		echo "$(GREEN) Common dependencies installed$(NC)"; \
	fi

.PHONY: deps
deps: install
	@echo "$(BLUE) Installing development dependencies...$(NC)"
	@$(PIP) install black flake8 pytest jupyter
	@echo "$(GREEN) Development dependencies installed$(NC)"

.PHONY: dataset-check
dataset-check:
	@echo "$(BLUE) Checking datasets for experiments...$(NC)"
	@mkdir -p $(TRAIN_DATASET_DIR) $(TEST_DATASET_DIR)
	@echo ""
	@echo "$(CYAN) Training Dataset (contact-high-school):$(NC)"
	@if [ -f "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt" ] && \
	   [ -f "$(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt" ]; then \
		echo "  $(GREEN) Required files found$(NC)"; \
		NODES=$$(wc -l < "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt" 2>/dev/null || echo "0"); \
		EDGES=$$(wc -l < "$(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt" 2>/dev/null || echo "0"); \
		echo "     Nodes: $$NODES, Hyperedges: $$EDGES"; \
	else \
		echo "  $(RED) Required files missing$(NC)"; \
		echo "     Expected: $(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt"; \
		echo "     Expected: $(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt"; \
	fi
	@echo ""
	@echo "$(CYAN) Testing Dataset (contact-primary-school):$(NC)"
	@if [ -f "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt" ] && \
	   [ -f "$(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		echo "  $(GREEN) Required files found$(NC)"; \
		NODES=$$(wc -l < "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt" 2>/dev/null || echo "0"); \
		EDGES=$$(wc -l < "$(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt" 2>/dev/null || echo "0"); \
		echo "     Nodes: $$NODES, Hyperedges: $$EDGES"; \
	else \
		echo "  $(RED)✗ Required files missing$(NC)"; \
		echo "     Expected: $(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt"; \
		echo "     Expected: $(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt"; \
	fi
	@echo ""
	@if [ ! -f "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt" ] || \
	   [ ! -f "$(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt" ] || \
	   [ ! -f "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt" ] || \
	   [ ! -f "$(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		echo "$(RED) DATASET ERROR: Missing required files$(NC)"; \
		echo "$(YELLOW) Please ensure all dataset files are in the correct directories$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN) All dataset files ready for experiments!$(NC)"; \
	fi

.PHONY: dataset-info
dataset-info: dataset-check
	@echo "$(BLUE) Dataset Information:$(NC)"
	@echo ""
	@echo "$(CYAN)Training Dataset (contact-high-school):$(NC)"
	@if [ -f "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt" ]; then \
		NODES=$$(wc -l < "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt"); \
		CLASSES=$$(sort "$(TRAIN_DATASET_DIR)/node-labels-contact-high-school.txt" | uniq | wc -l); \
		echo "   Nodes: $$NODES"; \
		echo "    Classes: $$CLASSES"; \
	fi
	@if [ -f "$(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt" ]; then \
		EDGES=$$(wc -l < "$(TRAIN_DATASET_DIR)/hyperedges-contact-high-school.txt"); \
		echo "   Hyperedges: $$EDGES"; \
	fi
	@echo ""
	@echo "$(CYAN)Testing Dataset (contact-primary-school):$(NC)"
	@if [ -f "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt" ]; then \
		NODES=$$(wc -l < "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt"); \
		CLASSES=$$(sort "$(TEST_DATASET_DIR)/node-labels-contact-primary-school.txt" | uniq | wc -l); \
		echo "   Nodes: $$NODES"; \
		echo "    Classes: $$CLASSES"; \
	fi
	@if [ -f "$(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		EDGES=$$(wc -l < "$(TEST_DATASET_DIR)/hyperedges-contact-primary-school.txt"); \
		echo "  🔗 Hyperedges: $$EDGES"; \
	fi

.PHONY: run
run: venv install dataset-check
	@echo "$(BLUE) Running HypergraphGRAND experiments...$(NC)"
	@echo "$(CYAN) Enhanced training with MLflow tracking$(NC)"
	@mkdir -p $(LOG_DIR) $(SAVED_MODELS_DIR)
	@$(PYTHON_VENV) train.py
	@echo "$(GREEN) Experiments completed! View results with: make mlflow$(NC)"

.PHONY: test
test: venv install
	@echo "$(BLUE)🧪 Testing model components...$(NC)"
	@$(PYTHON_VENV) -c "from model import HypergraphGRAND; print('✓ Model imported successfully')"
	@$(PYTHON_VENV) -c "import torch; print('✓ PyTorch available')"
	@$(PYTHON_VENV) -c "import mlflow; print('✓ MLflow available')"
	@echo "$(GREEN) All components tested successfully$(NC)"

.PHONY: validate
validate: venv install
	@echo "$(BLUE) Validating project structure...$(NC)"
	@for file in model.py train.py; do \
		if [ -f "$$file" ]; then \
			echo "  $(GREEN) $$file$(NC)"; \
		else \
			echo "  $(RED) $$file$(NC)"; \
		fi; \
	done
	@for dir in datasets logs $(SAVED_MODELS_DIR) venv; do \
		if [ -d "$$dir" ]; then \
			echo "  $(GREEN) $$dir/$(NC)"; \
		else \
			echo "  $(RED) $$dir/$(NC)"; \
		fi; \
	done
	@$(MAKE) test

.PHONY: mlflow
mlflow: venv install
	@echo "$(BLUE) Starting MLflow UI server...$(NC)"
	@echo "$(YELLOW) MLflow UI available at: http://localhost:5000$(NC)"
	@echo "$(YELLOW) Press Ctrl+C to stop the server$(NC)"
	@$(VENV_NAME)/bin/mlflow ui

.PHONY: mlflow-logs
mlflow-logs:
	@echo "$(BLUE) MLflow Experiment Results:$(NC)"
	@if [ -d "mlruns" ]; then \
		echo "$(GREEN) MLflow runs directory found$(NC)"; \
		RUNS=$$(find mlruns -name "meta.yaml" | wc -l); \
		echo "   Total runs: $$RUNS"; \
		echo "   Latest experiments:"; \
		find mlruns -name "meta.yaml" -exec dirname {} \; | sort | tail -5 | while read dir; do \
			echo "     $$dir"; \
		done; \
		echo "$(CYAN) Use 'make mlflow' to view detailed results in the UI$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No MLflow runs found. Run experiments first with 'make run'$(NC)"; \
	fi

.PHONY: mlflow-clean
mlflow-clean:
	@echo "$(BLUE)🧹 Cleaning MLflow runs and artifacts...$(NC)"
	@if [ -d "mlruns" ]; then \
		rm -rf mlruns; \
		echo "$(GREEN)✓ MLflow runs cleaned$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No MLflow runs to clean$(NC)"; \
	fi

.PHONY: models
models:
	@echo "$(BLUE) Saved Models:$(NC)"
	@if [ -d "$(SAVED_MODELS_DIR)" ]; then \
		if [ "$$(ls -A $(SAVED_MODELS_DIR))" ]; then \
			echo "$(GREEN) Models in $(SAVED_MODELS_DIR):$(NC)"; \
			ls -la $(SAVED_MODELS_DIR)/; \
		else \
			echo "$(YELLOW) $(SAVED_MODELS_DIR) is empty$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW) No saved models directory found$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN) MLflow also saves models - check with 'make mlflow'$(NC)"

.PHONY: logs
logs:
	@echo "$(BLUE) Recent experiment logs:$(NC)"
	@if [ -d "$(LOG_DIR)" ]; then \
		echo "$(GREEN) Log directory contents:$(NC)"; \
		if [ "$$(ls -A $(LOG_DIR))" ]; then \
			ls -la $(LOG_DIR)/ | head -10; \
		else \
			echo "$(YELLOW) $(LOG_DIR) is empty$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW) No log directory found$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN) MLflow logs: use 'make mlflow-logs' or 'make mlflow'$(NC)"

.PHONY: status
status:
	@echo "$(BLUE) Project Status:$(NC)"
	@echo ""
	@echo "$(CYAN) Environment:$(NC)"
	@if [ -d "$(VENV_NAME)" ]; then \
		echo "  $(GREEN) Virtual environment: $(VENV_NAME)$(NC)"; \
	else \
		echo "  $(RED) Virtual environment missing$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN) Project Structure:$(NC)"
	@ls -la . | grep -E "(model\.py|train\.py|datasets|logs|mlruns|$(SAVED_MODELS_DIR))" | while read line; do \
		echo "   $$line"; \
	done
	@echo ""
	@$(MAKE) dataset-check

.PHONY: lint
lint: venv install
	@echo "$(BLUE) Running code linting...$(NC)"
	@$(VENV_NAME)/bin/flake8 --max-line-length=88 --ignore=E203,W503 *.py 2>/dev/null || \
		echo "$(YELLOW)⚠️  flake8 not installed. Install with: make deps$(NC)"

.PHONY: format
format: venv deps
	@echo "$(BLUE) Formatting code...$(NC)"
	@$(VENV_NAME)/bin/black --line-length=88 *.py
	@echo "$(GREEN) Code formatted$(NC)"

.PHONY: freeze
freeze: venv
	@echo "$(BLUE) Generating requirements.txt...$(NC)"
	@$(PIP) freeze > requirements.txt
	@echo "$(GREEN) Requirements frozen to requirements.txt$(NC)"

.PHONY: clean
clean:
	@echo "$(BLUE) Cleaning generated files and cache...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.png" -type f -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN) Cleaned generated files and cache$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE) Cleaning everything...$(NC)"
	@rm -rf $(VENV_NAME)
	@rm -rf $(LOG_DIR)
	@rm -rf mlruns
	@rm -rf $(SAVED_MODELS_DIR)
	@echo "$(GREEN) Everything cleaned$(NC)"

.PHONY: check-venv
check-venv:
	@if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "$(GREEN) Virtual environment is activated$(NC)"; \
		echo "   Active environment: $$VIRTUAL_ENV"; \
	else \
		echo "$(YELLOW)⚠️  Virtual environment is not activated$(NC)"; \
		echo "   To activate: source $(VENV_NAME)/bin/activate"; \
	fi
