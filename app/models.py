from typing import List, Optional

from uuid import UUID, uuid4

from datetime import datetime

from sqlmodel import Field, SQLModel, Relationship, JSON

# --- 1. User table (Tenant) ---

class User(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    username: str

    created_at: datetime = Field(default_factory=datetime.utcnow)

    

    # Reverse relationship: One user has multiple Workflows

    workflows: List["Workflow"] = Relationship(back_populates="user")

# --- 2. Workflow table (Workflow/DAG) ---

class Workflow(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    user_id: UUID = Field(foreign_key="user.id") # Related to user

    status: str = Field(default="PENDING")       # PENDING, RUNNING, COMPLETED, FAILED

    created_at: datetime = Field(default_factory=datetime.utcnow)

    

    user: User = Relationship(back_populates="workflows")

    jobs: List["Job"] = Relationship(back_populates="workflow")

# --- 3. Job table (Job) ---

class Job(SQLModel, table=True):

    id: UUID = Field(default_factory=uuid4, primary_key=True)

    workflow_id: UUID = Field(foreign_key="workflow.id")

    

    name: str           # Task name, e.g. "Segmentation_Tile_1"

    job_type: str       # Task type, e.g. "inference", "tiling"

    

    # --- Core design: Branch-Aware ---

    # We use branch_id to identifywhich branch this task belongs to

    branch_id: str      # e.g. "branch_A", "main_branch"

    

    # --- Core design: DAG dependencies ---

    # Store list of parent task IDs, for building DAG graph

    # Will be stored as JSON format in PostgreSQL: ["uuid-1", "uuid-2"]

    parent_ids_json: str = Field(default="[]", sa_type=JSON) 

    

    status: str = Field(default="PENDING") # PENDING, QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED

    created_at: datetime = Field(default_factory=datetime.utcnow)  # for FIFO ordering

    # --- Failure/retry mechanism (Branch-Local) ---
    attempt: int = Field(default=0)  # Current retry count
    max_retries: int = Field(default=2)  # Maximum retry count
    error: Optional[str] = Field(default=None)  # Error information

    # --- ✨ Progress tracking fields ---
    total_tiles: int = Field(default=0)      # Total tile count
    processed_tiles: int = Field(default=0)  # Processed tile count

    # Store results (e.g. InstanSeg processed mask path, JSON format string)
    result_metadata: Optional[str] = Field(default=None)  # Changed to plain string to store JSON

    # --- Custom image path ---
    image_path: Optional[str] = Field(default=None)  # Custom image path, If empty, use default value

    workflow: Workflow = Relationship(back_populates="jobs")

