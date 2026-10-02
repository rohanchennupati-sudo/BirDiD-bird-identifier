from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

limiter = Limiter(key_func=get_remote_address)

# Read once at import time from RATE_LIMIT_PER_MINUTE (default 20).
PREDICT_RATE_LIMIT = f"{get_settings().rate_limit_per_minute}/minute"
