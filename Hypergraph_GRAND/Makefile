# HypergraphGRAND Makefile
# Manages virtual environment, dependencies, and common tasks

VENV_NAME = venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python
DATASET_DIR = datasets/contact-primary-school
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
	@echo "$(BLUE)Starting HypergraphGRAND workflow...$(NC)"
	@echo "$(BLUE)Step 1: Checking virtual environment...$(NC)"
	@if [ ! -d "$(VENV_NAME)" ]; then \
		echo "$(YELLOW)Virtual environment not found. Creating...$(NC)"; \
		$(MAKE) venv; \
		$(MAKE) install; \
	else \
		echo "$(GREEN)Virtual environment exists$(NC)"; \
	fi
	@echo "$(BLUE)Step 2: Checking dataset...$(NC)"
	@if [ ! -f "$(DATASET_DIR)/node-labels-contact-primary-school.txt" ] || \
	   [ ! -f "$(DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		echo "$(RED)ERROR: Dataset files are missing!$(NC)"; \
		echo "$(YELLOW)Required files:$(NC)"; \
		echo "  $(DATASET_DIR)/node-labels-contact-primary-school.txt"; \
		echo "  $(DATASET_DIR)/hyperedges-contact-primary-school.txt"; \
		echo "$(RED)Please download and place these files in the correct location.$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)Dataset files found$(NC)"; \
		wc -l $(DATASET_DIR)/*.txt; \
	fi
	@echo "$(BLUE)Step 3: Running main.py...$(NC)"
	@mkdir -p $(LOG_DIR)
	@$(PYTHON_VENV) main.py
	@echo "$(GREEN)Workflow completed successfully!$(NC)"

.PHONY: help
help:
	@echo "$(BLUE)HypergraphGRAND Project Makefile$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Default Command:$(NC)"
	@echo "  make             - Check venv, check dataset, run main.py (default)"
	@echo "  make default     - Same as above"
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
	@echo "  make clean       - Clean generated files"
	@echo "  make clean-all   - Clean everything including venv"
	@echo "  make lint        - Run code linting"
	@echo "  make format      - Format code with black"
	@echo ""
	@echo "$(GREEN)Utility Commands:$(NC)"
	@echo "  make logs        - View recent training logs"

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
dataset: dataset-check
	@if [ ! -f "$(DATASET_DIR)/node-labels-contact-primary-school.txt" ]; then \
		echo "$(BLUE)Downloading dataset...$(NC)"; \
		mkdir -p $(DATASET_DIR); \
		echo "$(YELLOW)Note: You need to manually download the dataset files:$(NC)"; \
		echo "  - node-labels-contact-primary-school.txt"; \
		echo "  - hyperedges-contact-primary-school.txt"; \
		echo "$(YELLOW)Place them in: $(DATASET_DIR)/$(NC)"; \
	else \
		echo "$(GREEN)Dataset files found$(NC)"; \
	fi

.PHONY: dataset-check
dataset-check:
	@echo "$(BLUE)Checking dataset...$(NC)"
	@mkdir -p $(DATASET_DIR)
	@if [ -f "$(DATASET_DIR)/node-labels-contact-primary-school.txt" ] && \
	   [ -f "$(DATASET_DIR)/hyperedges-contact-primary-school.txt" ]; then \
		echo "$(GREEN)Dataset files found$(NC)"; \
		wc -l $(DATASET_DIR)/*.txt; \
	else \
		echo "$(RED)Dataset files missing$(NC)"; \
		echo "$(YELLOW)Expected files:$(NC)"; \
		echo "  $(DATASET_DIR)/node-labels-contact-primary-school.txt"; \
		echo "  $(DATASET_DIR)/hyperedges-contact-primary-school.txt"; \
	fi

# Running targets
.PHONY: run
run: venv install dataset-check
	@echo "$(BLUE)Running HypergraphGRAND training...$(NC)"
	@mkdir -p $(LOG_DIR)
	@$(PYTHON_VENV) main.py

.PHONY: test
test: venv install
	@echo "$(BLUE)Running model tests...$(NC)"
	@$(PYTHON_VENV) model.py

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
	@echo "$(BLUE)Recent training logs:$(NC)"
	@if [ -d "$(LOG_DIR)" ]; then \
		ls -la $(LOG_DIR)/ | head -10; \
		echo ""; \
		echo "$(YELLOW)Latest log file:$(NC)"; \
		find $(LOG_DIR) -name "*.log" -type f -exec ls -t {} + | head -1 | xargs tail -20; \
	else \
		echo "$(YELLOW)No log directory found$(NC)"; \
	fi

.PHONY: clean
clean:
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.log" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "$(GREEN)Cleaned generated files$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo "$(BLUE)Cleaning everything...$(NC)"
	@rm -rf $(VENV_NAME)
	@rm -rf $(LOG_DIR)
	@echo "$(GREEN)Everything cleaned$(NC)"

.PHONY: check-venv
check-venv:
	@if [ "$$VIRTUAL_ENV" != "" ]; then \
		echo "$(GREEN)Virtual environment is activated$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment is not activated$(NC)"; \
		echo "To activate: source $(VENV_NAME)/bin/activate"; \
	fi
