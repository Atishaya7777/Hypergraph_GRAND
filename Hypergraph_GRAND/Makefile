# HypergraphGRAND Clustering Makefile
# Manages virtual environment, dependencies, and clustering experiments
# Train on contact-primary-school, test on contact-high-school

VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python
TRAIN_DATASET_DIR = datasets/contact-primary-school
TEST_DATASET_DIR = datasets/contact-high-school
LOG_DIR = logs
REQUIREMENTS_FILE = requirements.txt

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
	@echo "$(BLUE)Starting HypergraphGRAND clustering workflow...$(NC)"
	@echo "$(BLUE)Training on primary-school, testing on high-school datasets$(NC)"
	@echo "$(BLUE)Step 1: Checking virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Virtual environment not found. Creating...$(NC)"; \
		$(MAKE) venv; \
		$(MAKE) install; \
	else \
		echo "$(GREEN)Virtual environment exists$(NC)"; \
	fi
	@echo "$(BLUE)Step 2: Checking datasets...$(NC)"
	@$(MAKE) dataset-check
	@echo "$(BLUE)Step 3: Running clustering experiments...$(NC)"
	@mkdir -p $(LOG_DIR)
	@$(PYTHON_VENV) main.py
	@echo "$(GREEN)Clustering experiments completed successfully!$(NC)"

.PHONY: help
help:
	@echo "$(BLUE)HypergraphGRAND Clustering Project Makefile$(NC)"
	@echo "============================================="
	@echo ""
	@echo "$(CYAN) Clustering Task: Train on primary-school, test on high-school$(NC)"
	@echo ""
	@echo "$(GREEN)Default Command:$(NC)"
	@echo "  make             - Check venv, check datasets, run clustering experiments"
	@echo "  make default     - Same as above"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo ""
	@echo "$(GREEN)Running Commands:$(NC)"
	@echo "  make run         - Run clustering experiments (hidden dims 4, 8, 16)"
	@echo "  make test        - Test model components"
	@echo ""
	@echo "$(GREEN)Dataset Commands:$(NC)"
	@echo "  make dataset-check - Check both training and testing datasets"
	@echo "  make dataset     - Guide for dataset setup"
	@echo ""
	@echo "$(GREEN)MLflow Commands:$(NC)"
	@echo "  make mlflow      - Start MLflow UI server"
	@echo "  make mlflow-logs - View MLflow experiment results"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@echo "  make venv        - Create virtual environment"
	@echo "  make install     - Install Python dependencies"
	@echo ""
	@echo "$(GREEN)Running Commands:$(NC)"
	@echo "  make run         - Run the main training script"
	@echo "  make test        - Run model tests"
	@echo ""
	@echo "$(GREEN)Maintenance Commands:$(NC)"
	@echo "  make clean       - Clean generated files and cache"
	@echo "  make clean-all   - Clean everything including venv"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with black"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@echo "  make logs        - View recent training logs"
	@echo "  make freeze      - Update requirements.txt"
	@echo "  make check-venv  - Check virtual environment status"

.PHONY: venv
venv:
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		$(PYTHON) -m venv $(VENV_NAME); \
		echo "$(GREEN)Virtual environment created$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment already exists$(NC)"; \
	fi

.PHONY: install
install: venv 
	@echo "$(BLUE)Installing Python dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed$(NC)"

.PHONY: dataset
dataset: 
	@echo "$(BLUE)Dataset Setup Guide$(NC)"
	@echo "==================="
	@echo ""
	@echo "$(CYAN)For clustering experiments, you need both datasets:$(NC)"
	@echo ""
	@echo "$(YELLOW)Training Dataset (contact-primary-school):$(NC)"
	@echo "  📁 $(TRAIN_DATASET_DIR)/"
	@echo "    - node-labels-contact-primary-school.txt"
	@echo "    - hyperedges-contact-primary-school.txt"
	@echo ""
	@echo "$(YELLOW)Testing Dataset (contact-high-school):$(NC)"
	@echo "  📁 $(TEST_DATASET_DIR)/"
	@echo "    - node-labels-contact-high-school.txt"
	@echo "    - hyperedges-contact-high-school.txt"
	@echo ""
	@mkdir -p $(TRAIN_DATASET_DIR) $(TEST_DATASET_DIR)
	@$(MAKE) dataset-check

.PHONY: dataset-check
dataset-check:
	@echo "$(BLUE)Checking datasets for clustering...$(NC)"
	@mkdir -p $(TRAIN_DATASET_DIR) $(TEST_DATASET_DIR)
	@echo ""
	@echo "$(CYAN)Training Dataset (primary-school):$(NC)"
	@if [ -f "$(TRAIN_DATASET_DIR)/node-labels-contact-primary-school.txt" ] && \
	   [ -f "$(TRAIN_DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		echo "  $(GREEN)✓ Files found$(NC)"; \
		echo "     Lines: $$(wc -l $(TRAIN_DATASET_DIR)/*.txt 2>/dev/null | tail -1 | awk '{print $$1}' || echo '0') total"; \
	else \
		echo "  $(RED)✗ Files missing$(NC)"; \
		echo "    Expected: $(TRAIN_DATASET_DIR)/node-labels-contact-primary-school.txt"; \
		echo "    Expected: $(TRAIN_DATASET_DIR)/hyperedges-contact-primary-school.txt"; \
		MISSING_TRAIN=1; \
	fi
	@echo ""
	@echo "$(CYAN)Testing Dataset (high-school):$(NC)"
	@if [ -f "$(TEST_DATASET_DIR)/node-labels-contact-high-school.txt" ] && \
	   [ -f "$(TEST_DATASET_DIR)/hyperedges-contact-high-school.txt" ]; then \
		echo "  $(GREEN)✓ Files found$(NC)"; \
		echo "     Lines: $$(wc -l $(TEST_DATASET_DIR)/*.txt 2>/dev/null | tail -1 | awk '{print $$1}' || echo '0') total"; \
	else \
		echo "  $(RED)✗ Files missing$(NC)"; \
		echo "    Expected: $(TEST_DATASET_DIR)/node-labels-contact-high-school.txt"; \
		echo "    Expected: $(TEST_DATASET_DIR)/hyperedges-contact-high-school.txt"; \
		MISSING_TEST=1; \
	fi
	@echo ""
	@if [ ! -f "$(TRAIN_DATASET_DIR)/node-labels-contact-primary-school.txt" ] || \
	   [ ! -f "$(TRAIN_DATASET_DIR)/hyperedges-contact-primary-school.txt" ] || \
	   [ ! -f "$(TEST_DATASET_DIR)/node-labels-contact-high-school.txt" ] || \
	   [ ! -f "$(TEST_DATASET_DIR)/hyperedges-contact-high-school.txt" ]; then \
		echo "$(RED)❌ DATASET ERROR: Missing required files for clustering experiments$(NC)"; \
		echo "$(YELLOW)💡 Please download and place the dataset files in the correct directories$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN) All dataset files ready for clustering experiments!$(NC)"; \
	fi

# Running targets
.PHONY: run
run: venv install dataset-check
	@echo "$(BLUE)Running HypergraphGRAND clustering experiments...$(NC)"
	@echo "$(CYAN)🧠 Testing hidden dimensions: 4, 8, 16$(NC)"
	@mkdir -p $(LOG_DIR)
	@$(PYTHON_VENV) main.py

.PHONY: test
test: venv install
	@echo "$(BLUE)Testing model components...$(NC)"
	@$(PYTHON_VENV) -c "from model import clustering_loss_function, clustering_error_function; print(' Model components imported successfully')"

# MLflow targets
.PHONY: mlflow
mlflow: venv install
	@echo "$(BLUE)Starting MLflow UI server...$(NC)"
	@echo "$(YELLOW) MLflow UI will be available at: http://localhost:5000$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to stop the server$(NC)"
	@$(VENV_NAME)/bin/mlflow ui

.PHONY: mlflow-logs
mlflow-logs:
	@echo "$(BLUE)MLflow Experiment Results:$(NC)"
	@if [ -d "mlruns" ]; then \
		echo "$(GREEN)📁 MLflow runs directory found$(NC)"; \
		find mlruns -name "*.txt" -o -name "*.yaml" | head -5 | while read f; do \
			echo "  📄 $$f"; \
		done; \
		echo "$(CYAN)💡 Use 'make mlflow' to view detailed results in the UI$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  No MLflow runs found. Run experiments first with 'make run'$(NC)"; \
	fi

.PHONY: lint
lint: venv install
	@echo "$(BLUE)Running code linting...$(NC)"
	@$(VENV_NAME)/bin/flake8 --max-line-length=88 --ignore=E203,W503 *.py || true

.PHONY: format
format: venv 
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(VENV_NAME)/bin/black --line-length=88 *.py
	@echo "$(GREEN)Code formatted$(NC)"

.PHONY: freeze
freeze: venv
	@echo "$(BLUE)Generating requirements.txt from current environment...$(NC)"
	@$(PIP) freeze > requirements.txt
	@echo "$(GREEN)Requirements frozen$(NC)"

.PHONY: logs
logs:
	@echo "$(BLUE)Recent clustering experiment logs:$(NC)"
	@if [ -d "$(LOG_DIR)" ]; then \
		echo "$(GREEN)📁 Log directory contents:$(NC)"; \
		ls -la $(LOG_DIR)/ | head -10; \
		echo ""; \
		echo "$(YELLOW)Latest log file:$(NC)"; \
		find $(LOG_DIR) -name "*.log" -type f -exec ls -t {} + 2>/dev/null | head -1 | xargs tail -20 2>/dev/null || echo "No log files found"; \
	else \
		echo "$(YELLOW)📁 No log directory found$(NC)"; \
	fi
	@echo ""
	@echo "$(CYAN)💡 MLflow logs: use 'make mlflow-logs' or 'make mlflow'$(NC)"

.PHONY: clean
clean:
	@echo "$(BLUE)Cleaning generated files and cache...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf saved_models/*.pth 2>/dev/null || true
	@echo "$(GREEN) Cleaned generated files and cache$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE)Cleaning everything (including venv and MLflow)...$(NC)"
	@rm -rf $(VENV_NAME)
	@rm -rf $(LOG_DIR)
	@rm -rf mlruns 2>/dev/null || true
	@rm -rf saved_models 2>/dev/null || true
	@echo "$(GREEN) Everything cleaned$(NC)"

.PHONY: check-venv
check-venv:
	@if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "$(GREEN)Virtual environment is activated$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment is not activated$(NC)"; \
		echo "To activate: source $(VENV_NAME)/bin/activate"; \
	fi
