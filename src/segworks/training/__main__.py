import logging
from pathlib import Path

from ..config_parsing import build_pipeline
from .train import train as main

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    import argparse

    from .logging import configure_logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML for training.",
    )
    args = parser.parse_args()
    pipeline = build_pipeline(args.config)

    logger.info(f"Starting training based on config file: {args.config}")

    logger = configure_logging(pipeline["output_path"])

    main(pipeline)
