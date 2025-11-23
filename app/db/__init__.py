from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

# 使用配置中的异步数据库 URL
# 注意：如果在 docker 容器内运行，POSTGRES_HOST 应该是 'db'
# 如果在本地运行，POSTGRES_HOST 应该是 'localhost'
DATABASE_URL = settings.async_database_url

engine = create_async_engine(DATABASE_URL, echo=True, future=True)


async def init_db():
    """初始化数据库，删除并重新创建所有表（开发模式）"""
    async with engine.begin() as conn:
        # 先删除所有表（注意：这会删除所有数据）
        await conn.run_sync(SQLModel.metadata.drop_all)
        # 然后创建所有表
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncSession:
    """获取异步数据库会话"""
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session() as session:
        yield session
