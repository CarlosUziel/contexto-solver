import os
import sys

from loguru import logger as loguru_logger
from loguru._logger import Logger  # Import Logger for type hinting


def setup_logger(
    name: str = __name__, level: str = "INFO"
) -> Logger:  # Use Logger for type hint
    """Sets up and returns a configured Loguru logger instance.

    The logger will output to both the console and a file named 'app.log'
    in the '.logs' directory. It configures Loguru's default logger.

    Args:
        name: The name for the logger (Loguru typically uses module name automatically).
        level: The logging level (e.g., "INFO", "DEBUG").
               Defaults to "INFO".

    Returns:
        A configured Loguru logger instance.
    """
    # Remove default Loguru handler to avoid duplicate console logs if any
    loguru_logger.remove()

    # Console Handler
    loguru_logger.add(
        sys.stdout,
        level=level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File Handler
    log_dir = ".logs"
    log_file_name = "app.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        loguru_logger.add(
            log_file_path,
            level=level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate log file when it reaches 10 MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress rotated files
            enqueue=True,  # Asynchronous logging
        )
    except Exception as e:
        # Use print for critical errors during logger setup
        print(
            f"Failed to set up file handler for Loguru logger: {e}",
            file=sys.stderr,
        )

    # Loguru logger is configured globally, so we just return it.
    # The 'name' argument is less critical for Loguru as it captures module context automatically.
    return loguru_logger


# Default application logger instance
app_logger = setup_logger("contexto_solver_app")
