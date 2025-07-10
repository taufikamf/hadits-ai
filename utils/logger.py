"""
Logging configuration for hadits-ai.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import settings


def setup_logging(log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional path to log file. If not provided, uses settings.log_file
    """
    # Create logs directory if not exists
    log_file = log_file or settings.log_file
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging format
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)
    
    # Add rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)
    
    # Add console handler if in debug mode
    if settings.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        root_logger.addHandler(console_handler)
    
    # Set logging levels for third-party libraries
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully") 