import os
import logging
import json
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class JSONFormatter(logging.Formatter):
    """Formats log records as strict JSON for the Durable Audit Trail."""
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "event": record.getMessage(),
            "namespace": record.name,
        }
        # Inject structured L7 contract details if provided
        if hasattr(record, "details"):
            log_record["details"] = record.details
        
        return json.dumps(log_record)

def setup_logging():
    # 1. Standard Human-Readable Console Output
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)-18s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()]
    )

    # 2. Machine-Readable JSON Audit Trail
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False  # Keep JSON out of the human console

    audit_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "audit_trail.jsonl"), 
        maxBytes=5_000_000, 
        backupCount=3
    )
    audit_handler.setFormatter(JSONFormatter())
    
    if not audit_logger.handlers:
        audit_logger.addHandler(audit_handler)

def get_logger(name: str):
    return logging.getLogger(name)

def get_audit_logger():
    return logging.getLogger("audit")