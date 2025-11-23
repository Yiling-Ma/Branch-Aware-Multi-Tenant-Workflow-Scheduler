from fastapi import FastAPI, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from app.db import init_db, get_session
from app.models import User, Workflow, Job
from uuid import UUID
from pydantic import BaseModel
from typing import List
import json

app = FastAPI()

# 启动时初始化数据库表
@app.on_event("startup")
async def on_startup():
    await init_db()

# --- API 1: 创建测试用户 ---
@app.post("/users/", response_model=User)
async def create_user(user: User, session: AsyncSession = Depends(get_session)):
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user

# --- API 2: 提交一个 Workflow (最重要的一步) ---
# 用户会发送一个包含多个 Job 的列表

# 定义接收前端请求的数据结构 (Schema)
class JobCreate(BaseModel):
    name: str
    job_type: str
    branch_id: str
    parent_indices: List[int] = []  # 这是一个技巧：引用列表中的第几个 job 作为父节点

class WorkflowCreate(BaseModel):
    user_id: UUID
    jobs: List[JobCreate]

@app.post("/workflows/")
async def create_workflow(workflow_data: WorkflowCreate, session: AsyncSession = Depends(get_session)):
    # 1. 创建 Workflow
    new_workflow = Workflow(user_id=workflow_data.user_id)
    session.add(new_workflow)
    await session.commit()
    await session.refresh(new_workflow)
    
    # 2. 创建 Jobs 并解析依赖关系
    # 因为 Job 还没存入数据库，没有 ID，我们先用 list index 暂存依赖
    created_jobs = []
    
    for index, job_in in enumerate(workflow_data.jobs):
        job = Job(
            workflow_id=new_workflow.id,
            name=job_in.name,
            job_type=job_in.job_type,
            branch_id=job_in.branch_id,
            status="PENDING"
        )
        session.add(job)
        # 暂时 flush 拿到 ID，但不 commit
        await session.flush()
        await session.refresh(job)
        created_jobs.append(job)
    
    # 3. 第二遍循环：填入 parent_ids (因为现在大家都有 ID 了)
    for index, job_in in enumerate(workflow_data.jobs):
        parent_uuids = []
        for parent_idx in job_in.parent_indices:
            if parent_idx < index:  # 防止循环依赖
                parent_uuids.append(str(created_jobs[parent_idx].id))
        
        # 更新数据库里的 Job
        created_jobs[index].parent_ids_json = json.dumps(parent_uuids)
        session.add(created_jobs[index])
        
    await session.commit()
    return {"workflow_id": new_workflow.id, "job_count": len(created_jobs)}

