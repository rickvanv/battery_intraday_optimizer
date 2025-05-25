import logging
import logging.handlers
import sys
import warnings
import os
from datetime import datetime
from cli import get_logger_args

# Default logger configuration
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
)
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def is_debug_mode():
    """Detect if the script is running in debug mode (PyCharm, VSCode)."""
    return any(
        debugger in sys.modules for debugger in ("pydevd", "debugpy", "ptvsd")
    )


def configure_logger(
    log_level,
    log_file_name,
    log_file=False,
    log_rotation=False,
    suppress_warnings=True,
):
    """
    Configure the root logger with console and optional file logging.

    Parameters
    ----------
    log_level : str, optional
        Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        Defaults to None, which falls back to INFO.
    log_file : bool, optional
        If True, enables logging to a file. Defaults to False.
    log_file_name : str, optional
        Base filename for log files. Defaults to PROJECT_LOG.
    log_rotation : bool, optional
        If True, enables rotating file handler (10MB per file, 5 backups).
        Defaults to False.
    suppress_warnings : bool, optional
        If True, suppresses runtime warnings. Defaults to True.

    Returns
    -------
    None

    Notes
    -----
    - Console logging is always enabled.
    - If `log_file` is enabled, logs are stored in the current working directory.
    - File logs are named using `log_file_name` with a timestamp suffix.
    - Rotation allows a maximum of 5 backup files, each with a 10MB limit.
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Set root logger
    root_logger = logging.getLogger()

    # Remove existing handlers to avoid duplicates
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Set root logger level
    root_logger.setLevel(numeric_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )
    root_logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        timestamp = datetime.now().strftime(DEFAULT_DATE_FORMAT)
        filepath = os.path.join(
            os.getcwd(), f"{log_file_name}_{timestamp}.log"
        )

        if log_rotation:
            file_handler = logging.handlers.RotatingFileHandler(
                filepath, maxBytes=10 * 1024 * 1024, backupCount=5
            )
        else:
            file_handler = logging.FileHandler(filepath)

        file_handler.setFormatter(
            logging.Formatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        root_logger.addHandler(file_handler)

    # Suppress warnings if specified
    if suppress_warnings:
        warnings.filterwarnings("ignore", category=RuntimeWarning)


def setup_logger():
    """
    Set up the logger based on debug mode or command-line arguments.
    """
    # Detect debug mode
    debug_mode = is_debug_mode()
    log_level = "DEBUG" if debug_mode else "INFO"

    # Get command-line arguments from cli.py
    parser = get_logger_args()
    args, unknown = parser.parse_known_args()

    # Configure logger
    configure_logger(
        log_level=log_level,
        log_file=args.log_file,
        log_file_name=args.log_file_name,
        log_rotation=args.log_rotation,
        suppress_warnings=args.suppress_warnings,
    )
