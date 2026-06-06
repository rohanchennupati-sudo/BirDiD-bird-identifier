from app.config import get_settings

settings = get_settings()

print(settings.environment)
print(settings.max_image_size_mb)
print(settings.allowed_origins_list)