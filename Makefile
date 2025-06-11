VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: all install clean activate jupyter-kernel remove-kernel

all: install

install:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Environment setup complete."

jupyter-kernel:
	@echo "Registering Jupyter kernel as 'fairfluence'..."
	@$(PIP) install ipykernel
	@$(VENV_DIR)/bin/python -m ipykernel install --user --name=fairfluence --display-name "fairfluence"

remove-kernel:
	@echo "Removing old Jupyter kernel 'fairfluence' (if exists)..."
	@jupyter kernelspec uninstall -f fairfluence || true

activate:
	@echo "To activate the environment, run:"
	@echo "$(ACTIVATE)"

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Clean complete."
