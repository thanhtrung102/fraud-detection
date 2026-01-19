# Fraud Detection MLOps Project Makefile

.PHONY: help install install-dev test lint format type-check clean \
        train train-quick serve docker-build docker-push docker-up docker-down \
        mlflow-ui prefect-ui terraform-init terraform-plan terraform-apply

# Default target
help:
	@echo "Fraud Detection MLOps - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-cov       Run tests with coverage"
	@echo "  make lint           Run linting (ruff)"
	@echo "  make format         Format code (black)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "Training:"
	@echo "  make train          Run full training pipeline"
	@echo "  make train-quick    Run training without Optuna"
	@echo ""
	@echo "Serving:"
	@echo "  make serve          Start FastAPI server locally"
	@echo "  make mlflow-ui      Start MLflow UI"
	@echo "  make prefect-ui     Start Prefect UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make terraform-init   Initialize Terraform"
	@echo "  make terraform-plan   Plan infrastructure changes"
	@echo "  make terraform-apply  Apply infrastructure changes"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Clean build artifacts"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install -r requirements-dev.txt
	pre-commit install

# =============================================================================
# Development
# =============================================================================

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-cov:
	pytest tests/ -v --cov=src --cov=mlops --cov=pipelines --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ mlops/ pipelines/ tests/ deployment/

lint-fix:
	ruff check --fix src/ mlops/ pipelines/ tests/ deployment/

format:
	black src/ mlops/ pipelines/ tests/ deployment/

format-check:
	black --check src/ mlops/ pipelines/ tests/ deployment/

type-check:
	mypy src/ mlops/ pipelines/ --ignore-missing-imports

quality: lint format-check type-check test
	@echo "All quality checks passed!"

pre-commit:
	pre-commit run --all-files

# =============================================================================
# Training
# =============================================================================

train:
	python -m src.main

train-quick:
	python -m src.main --no-optuna

train-pipeline:
	python -m pipelines.training_pipeline

inference:
	python -m pipelines.inference_pipeline

monitoring:
	python -m pipelines.monitoring_pipeline

# =============================================================================
# Serving
# =============================================================================

serve:
	uvicorn deployment.api.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

prefect-ui:
	prefect server start

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t fraud-detection-api:latest -f deployment/Dockerfile .
	docker build -t fraud-detection-training:latest -f deployment/Dockerfile.training .

docker-up:
	docker-compose -f deployment/docker-compose.yml up -d

docker-down:
	docker-compose -f deployment/docker-compose.yml down

docker-logs:
	docker-compose -f deployment/docker-compose.yml logs -f

docker-push:
	@echo "Push to your container registry:"
	@echo "  docker tag fraud-detection-api:latest <registry>/fraud-detection-api:latest"
	@echo "  docker push <registry>/fraud-detection-api:latest"

# =============================================================================
# Infrastructure (Terraform)
# =============================================================================

terraform-init:
	cd infrastructure && terraform init

terraform-plan:
	cd infrastructure && terraform plan

terraform-apply:
	cd infrastructure && terraform apply

terraform-destroy:
	cd infrastructure && terraform destroy

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	rm -rf catboost_info/ 2>/dev/null || true
	@echo "Cleaned build artifacts"

clean-models:
	rm -rf models/*.joblib
	@echo "Cleaned model files"

clean-results:
	rm -rf results/*.png results/*.json results/*.html
	@echo "Cleaned result files"

clean-all: clean clean-models clean-results
	@echo "Cleaned all artifacts"
