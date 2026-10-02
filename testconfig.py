from app.config import get_settings

# Quick check that settings load from environment variables / .env.
# Run with `python testconfig.py`. With the defaults it prints:
#   development
#   10
#   20
settings = get_settings()

print(settings.environment)
print(settings.max_image_size_mb)
print(settings.rate_limit_per_minute)
