from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

# Use asynchronous database URL from configuration
# Note:If running in Docker container,POSTGRES_HOST should be 'db'
# If running locally,POSTGRES_HOST should be 'localhost'
DATABASE_URL = settings.async_database_url

engine = create_async_engine(DATABASE_URL, echo=False, future=True)


async def init_db():
    """
    Initialize database, create tables if they don't exist.
    This preserves existing data - tables are only created if they don't exist.
    For development/testing, use drop_all() manually if you need to reset the database.
    """
    try:
        print("🔄 Initializing database...")
        async with engine.begin() as conn:
            # Only create tables if they don't exist (preserves existing data)
            print("   ✅ Creating tables if they don't exist...")
            await conn.run_sync(SQLModel.metadata.create_all)
        print("✅ Database initialization complete (existing data preserved)")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def get_session() -> AsyncSession:
    """Get asynchronous database session"""
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session

