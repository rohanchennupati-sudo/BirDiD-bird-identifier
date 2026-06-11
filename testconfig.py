from app.config import get_settings

settings = get_settings()

print(settings.environment)
print(settings.max_image_size_mb)
print(settings.allowed_origins_list)
# This file is for testing that the configuration loads correctly and that environment variables are read as expected.
#  Run it with `python testconfig.py` to see the output.
# Output should be 10 and development