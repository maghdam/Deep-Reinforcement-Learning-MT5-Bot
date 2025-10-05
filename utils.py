import datetime

def log_and_print(message, is_error=False):
    """Prints a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_level = "ERROR" if is_error else "INFO"
    print(f"[{timestamp}] [{log_level}] {message}")
