"""
Distill + Align: End-to-end LLM alignment pipeline.

Modules:
    config/      - YAML configuration management and validation
    data/        - Dataset loading, preprocessing, quality validation
    training/    - SFT, DPO, and RLHF trainer wrappers
    evaluation/  - Model benchmarking and comparison
    serving/     - vLLM inference engine, FastAPI gateway
    monitoring/  - Prometheus metrics
    tracking/    - Experiment tracking
"""

__version__ = "0.1.0"
