# Variables
PYTHON=python
SCRIPT=main.py

# Default task
.PHONY: all
all: help

# Prepare data
.PHONY: prepare
prepare:
	$(PYTHON) $(SCRIPT) prepare --file_path mm-5G.csv

# Train model
.PHONY: train
train:
	$(PYTHON) $(SCRIPT) train

# Predict with saved model
.PHONY: predict
predict:
	$(PYTHON) $(SCRIPT) predict

# Save model
.PHONY: save
save:
	$(PYTHON) $(SCRIPT) save

# Load and test model
.PHONY: load
load:
	$(PYTHON) $(SCRIPT) load

# Clean up saved files
.PHONY: clean
clean:
	rm -f processed_data.pkl throughput_model.keras throughput_model.pkl scaler_*.pkl
	echo "All saved files removed."

# CI: Lint with flake8 (restricted to model_pipeline.py and main.py)
.PHONY: lint
lint:
	flake8 model_pipeline.py main.py

# CI: Format with black and isort (restricted to model_pipeline.py and main.py)
.PHONY: format
format:
	black model_pipeline.py main.py && isort model_pipeline.py main.py

# CI: Security check with bandit (restricted to model_pipeline.py and main.py)
.PHONY: security
security:
	bandit -r model_pipeline.py main.py

# CI: Run tests with pytest (restricted to model_pipeline.py and main.py)
.PHONY: test
test:
	pytest model_pipeline.py main.py

# Start FastAPI server
.PHONY: api
api:
	@echo "Starting FastAPI server on port 8000..."
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload > api.log 2>&1 &

# Start Flask server
.PHONY: web
web:
	@echo "Starting Flask server on port 5001..."
	python web.py > web.log 2>&1 &

# Run all CI checks
.PHONY: ci
ci: lint format security test

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f --tail=100

push-back:
	docker tag pi_backend ines659/pi:backend
	docker push ines659/pi:backend

push-front:
	docker tag pi_frontend ines659/pi:frontend
	docker push ines659/pi:frontend

deploy: build up logs

# Display available commands
.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  prepare     Prepare the dataset"
	@echo "  train       Train the model"
	@echo "  predict     Make predictions on the test set"
	@echo "  save        Save the trained model"
	@echo "  load        Load and test the model"
	@echo "  clean       Remove all saved files"
	@echo "  lint        Check code quality with flake8 (model_pipeline.py, main.py)"
	@echo "  format      Format code with black and isort (model_pipeline.py, main.py)"
	@echo "  security    Check security with bandit (model_pipeline.py, main.py)"
	@echo "  test        Run tests with pytest (model_pipeline.py, main.py)"
	@echo "  ci          Run all CI checks"
	@echo "  api         Start the FastAPI server on port 8000"
	@echo "  web         Start the Flask server on port 5001"