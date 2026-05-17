"""
Distill + Align: End-to-end LLM alignment pipeline.

Modules:
    config/      - YAML configuration management and validation
    data/        - Dataset loading, preprocessing, Glue ETL, quality validation
    training/    - SFT, DPO, and RLHF trainer wrappers
    evaluation/  - Model benchmarking and comparison reports
    serving/     - vLLM inference engine, FastAPI gateway, SageMaker endpoints
    monitoring/  - Prometheus metrics (local) and CloudWatch alarms (AWS)
    tracking/    - Weights & Biases experiment tracking
"""

__version__ = "0.1.0"
