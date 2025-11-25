"""
Redis-based Token Bucket Rate Limiting
"""
import time
import json
from typing import Optional
import redis.asyncio as redis
from app.core.config import settings


class TokenBucket:
    """
    Redis-based token bucket rate limiter
    
    Algorithm:
    - Each key has a bucket with max capacity
    - Tokens are added at a fixed rate (refill_rate per second)
    - Request consumes 1 token if available, otherwise denied
    """
    
    def __init__(self, redis_client: redis.Redis, capacity: int, refill_rate: float):
        """
        Args:
            redis_client: Redis async client
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
    
    async def acquire(self, key: str, tokens: int = 1) -> bool:
        """
        Try to acquire tokens from bucket
        
        Args:
            key: Redis key for this bucket
            tokens: Number of tokens to consume (default: 1)
        
        Returns:
            True if tokens available and consumed, False otherwise
        """
        now = time.time()
        
        # Lua script for atomic token bucket operation
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- Get current bucket state
        local bucket_data = redis.call('GET', key)
        local tokens, last_refill = 0, now
        
        if bucket_data then
            local data = cjson.decode(bucket_data)
            tokens = data.tokens
            last_refill = data.last_refill
        end
        
        -- Refill tokens based on time elapsed
        local elapsed = now - last_refill
        local tokens_to_add = elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- Check if enough tokens available
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            local new_data = cjson.encode({
                tokens = tokens,
                last_refill = now
            })
            redis.call('SET', key, new_data)
            redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity
            return 1  -- Allowed
        else
            -- Update last_refill even if denied (for accurate refill calculation)
            local new_data = cjson.encode({
                tokens = tokens,
                last_refill = now
            })
            redis.call('SET', key, new_data)
            redis.call('EXPIRE', key, 3600)
            return 0  -- Denied
        end
        """
        
        try:
            result = await self.redis.eval(
                lua_script,
                1,  # Number of keys
                key,
                self.capacity,
                self.refill_rate,
                tokens,
                now
            )
            return bool(result)
        except Exception as e:
            print(f"Rate limit error for key {key}: {e}")
            # On error, allow request (fail open)
            return True
    
    async def get_tokens_available(self, key: str) -> float:
        """Get current number of tokens available (for monitoring)"""
        now = time.time()
        
        try:
            bucket_data = await self.redis.get(key)
            if not bucket_data:
                return self.capacity
            
            data = json.loads(bucket_data)
            tokens = data.get('tokens', 0)
            last_refill = data.get('last_refill', now)
            
            # Refill tokens
            elapsed = now - last_refill
            tokens_to_add = elapsed * self.refill_rate
            tokens = min(self.capacity, tokens + tokens_to_add)
            
            return tokens
        except Exception as e:
            print(f"Error getting tokens for key {key}: {e}")
            return self.capacity


class RateLimiter:
    """
    Rate limiter with different limits for different endpoints
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Define rate limits per endpoint
        # Format: (capacity, refill_rate_per_second)
        # These are per-user limits (using user_id as identifier)
        self.limits = {
            'create_workflow': (5, 1.0),       # 5 requests capacity, refill 1/sec (max 5 req/sec burst, 1 req/sec sustained)
            'create_job': (20, 5.0),           # 20 requests, refill 5/sec
            'upload_file': (5, 1.0),           # 5 requests, refill 1/sec
            'export_results': (10, 2.0),       # 10 requests, refill 2/sec
            'default': (30, 10.0),             # Default: 30 requests, refill 10/sec
        }
        
        self.buckets = {}
        for endpoint, (capacity, refill_rate) in self.limits.items():
            self.buckets[endpoint] = TokenBucket(redis_client, capacity, refill_rate)
    
    async def check_rate_limit(self, endpoint: str, identifier: str = "global") -> bool:
        """
        Check if request is within rate limit
        
        Args:
            endpoint: Endpoint name (e.g., 'create_workflow')
            identifier: User ID or IP address for per-user/IP limiting
        
        Returns:
            True if allowed, False if rate limited
        """
        # Use endpoint-specific bucket or default
        bucket_name = endpoint if endpoint in self.buckets else 'default'
        bucket = self.buckets[bucket_name]
        
        # Create key: rate_limit:{endpoint}:{identifier}
        key = f"rate_limit:{endpoint}:{identifier}"
        
        allowed = await bucket.acquire(key)
        return allowed
    
    async def get_tokens_available(self, endpoint: str, identifier: str = "global") -> float:
        """Get available tokens for monitoring"""
        bucket_name = endpoint if endpoint in self.buckets else 'default'
        bucket = self.buckets[bucket_name]
        key = f"rate_limit:{endpoint}:{identifier}"
        return await bucket.get_tokens_available(key)


# Global rate limiter instance (will be initialized in main.py)
rate_limiter: Optional[RateLimiter] = None


async def init_rate_limiter():
    """Initialize global rate limiter"""
    global rate_limiter
    try:
        redis_client = redis.from_url(settings.redis_url)
        rate_limiter = RateLimiter(redis_client)
        print("Rate limiter initialized")
    except Exception as e:
        print(f"Failed to initialize rate limiter: {e}")
        rate_limiter = None


async def get_rate_limiter() -> Optional[RateLimiter]:
    """Get global rate limiter instance"""
    return rate_limiter

