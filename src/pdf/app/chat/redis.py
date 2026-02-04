import redis
from app.settings import get_settings

settings = get_settings()

client = redis.Redis.from_url(settings.redis_uri, decode_responses=True)