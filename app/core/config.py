from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # PostgreSQL configuration
    POSTGRES_USER: str = "user"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "scheduler_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Redis configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Scheduler configuration
    MAX_WORKERS: int = 4  # Global maximum concurrent worker count
    MAX_JOBS_PER_USER: int = 50  # Maximum concurrent jobs (PENDING+QUEUED+RUNNING) per user
    SCHEDULER_SEMAPHORE_SIZE: int = 10  # Maximum jobs scheduler can process per tick
    
    # Worker configuration
    INSTANSEG_MODEL_TYPE: str = "brightfield_nuclei"  # Optional: "brightfield_nuclei", "fluorescence_nuclei", "fluorescence_cells"
    INSTANSEG_IMAGE_READER: str = "openslide"  # for reading files: "openslide" or "tiffslide"
    INSTANSEG_PIXEL_SIZE: float = 0.5  # Pixel size(micrometers per pixel),has great impact on detection results,needs to be adjusted based on actual images
    
    @property
    def database_url(self) -> str:
        """Build PostgreSQL database connection URL (synchronous)"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def async_database_url(self) -> str:
        """Build PostgreSQL asynchronous database connection URL"""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def redis_url(self) -> str:
        """Build Redis connection URL"""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

