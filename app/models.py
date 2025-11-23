from typing import List, Optional

from uuid import UUID, uuid4

from datetime import datetime

from sqlmodel import Field, SQLModel, Relationship, JSON

# --- 1. 用户表 (Tenant) ---

class User(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    username: str

    created_at: datetime = Field(default_factory=datetime.utcnow)

    

    # 反向关系：一个用户有多个 Workflow

    workflows: List["Workflow"] = Relationship(back_populates="user")

# --- 2. 工作流表 (Workflow/DAG) ---

class Workflow(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    user_id: UUID = Field(foreign_key="user.id") # 关联到用户

    status: str = Field(default="PENDING")       # PENDING, RUNNING, COMPLETED, FAILED

    created_at: datetime = Field(default_factory=datetime.utcnow)

    

    user: User = Relationship(back_populates="workflows")

    jobs: List["Job"] = Relationship(back_populates="workflow")

# --- 3. 任务表 (Job) ---

class Job(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    workflow_id: UUID = Field(foreign_key="workflow.id")

    

    name: str           # 任务名称，例如 "Segmentation_Tile_1"

    job_type: str       # 任务类型，例如 "inference", "tiling"

    

    # --- 核心设计：Branch-Aware ---

    # 我们用 branch_id 来标识这个任务属于哪条分支

    branch_id: str      # 例如 "branch_A", "main_branch"

    

    # --- 核心设计：DAG 依赖 ---

    # 存储父任务 ID 的列表，用于构建 DAG 图

    # 在 PostgreSQL 中会存储为 JSON 格式: ["uuid-1", "uuid-2"]

    parent_ids_json: str = Field(default="[]", sa_type=JSON) 

    

    status: str = Field(default="PENDING") # PENDING, QUEUED, RUNNING, SUCCEEDED, FAILED

    

    # 存储结果 (比如 InstanSeg 处理完的 mask 路径)

    result_metadata: Optional[str] = Field(default=None, sa_type=JSON)

    

    workflow: Workflow = Relationship(back_populates="jobs")

