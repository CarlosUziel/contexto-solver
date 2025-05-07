import logging
import sys
import os  # Added import


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Sets up and returns a configured logger instance.

    The logger will output to both the console and a file named 'app.log'.
    It avoids adding multiple handlers if the logger instance has already
    been configured.

    Args:
        name: The name for the logger. Defaults to the module name.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
               Defaults to logging.INFO.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already has them
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler
        # Ensure the .logs directory exists
        log_dir = ".logs"
        log_file_name = "app.log"
        log_file_path = os.path.join(log_dir, log_file_name)

        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            fh = logging.FileHandler(log_file_path)  # Changed to use .logs/app.log
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            # Log to stderr if file handler setup fails, as logger might not be fully set up.
            print(
                f"Failed to set up file handler for logger '{name}': {e}",
                file=sys.stderr,
            )

    # Prevent log messages from propagating to the root logger
    logger.propagate = False

    return logger


# Default application logger instance
app_logger = setup_logger("contexto_solver_app")
