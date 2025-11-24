import asyncio
import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import redis.asyncio as redis
import json
from uuid import UUID
from sqlmodel import select
from app.db import engine
from app.models import Workflow, Job
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy import and_, or_
from sqlmodel.ext.asyncio.session import AsyncSession
from app.core.config import settings

REDIS_URL = settings.redis_url
MAX_ACTIVE_USERS = 3
MAX_WORKERS = settings.MAX_WORKERS

# --- Helper function:Check if branch is busy ---
async def is_branch_busy(session, branch_id):
    """
    Check if this branch has running tasks(global lock, across workflows)
    Note:only check RUNNING status,do not check QUEUED status
    because QUEUED tasks have not actually started running,should not block other tasks in the same branch
    Only RUNNING tasks will block other tasks in the same branch(serial execution)
    """
    statement = select(Job).where(
        Job.branch_id == branch_id,
        Job.status == "RUNNING"
    )
    result = await session.exec(statement)
    busy = result.first() is not None
    if busy:
        print(f"      🔒 Branch {branch_id} is busy (has RUNNING job)")
    return busy

# --- Helper function:Update user activity time ---
async def update_user_activity(r, user_id):
    """Update user last activity time(for timeout detection)"""
    import time
    await r.set(f"user_activity:{user_id}", str(time.time()), ex=600)  # 10minutes expiration

# --- Helper function:Ensure user is active or queued ---
async def ensure_user_active(r, user_id):
    """Ensure user is active or joins waiting queue"""
    if await r.sismember("active_users", user_id):
        # Update activity time
        await update_user_activity(r, user_id)
        return True
    
    count = await r.scard("active_users")
    active_users_list = await r.smembers("active_users")
    
    # Debug information:Show current active users
    if count > 0:
        print(f"      📋 Current active users ({count}/{MAX_ACTIVE_USERS}): {', '.join(list(active_users_list)[:3])}")
    
    if count < MAX_ACTIVE_USERS:
        await r.sadd("active_users", user_id)
        await update_user_activity(r, user_id)  # Record activity time
        print(f"✅  {user_id} Gained execution permission")
        return True
    
    # Inactive and no slot available:enter waiting queue(deduplicate)
    if not await r.lpos("waiting_users", user_id):
        await r.rpush("waiting_users", user_id)
        print(f"🕒  {user_id} Enter waiting queue (Active users full: {count}/{MAX_ACTIVE_USERS})")
    return False

# --- Helper function:Handle Job retry (Branch-Local) ---
async def retry_failed_job(session, job: Job):
    """
    Call this function for retry when worker fails
    Branch-Local: Failure only makes subsequent tasks in same branch wait,does not affect other branches
    """
    if job.status == "FAILED" and job.attempt < job.max_retries:
        job.attempt += 1
        job.status = "PENDING"
        job.error = None
        session.add(job)
        await session.commit()
        print(f"🔁 Job {job.name} retry {job.attempt}/{job.max_retries} (Branch: {job.branch_id})")
        return True
    return False

# --- Helper function:Release user and activate next waiting user ---
async def release_user_if_done(session, r, user_id):
    """Check if user still has pending tasks,If not, release and activate next waiting user"""
    # Query through workflow if this user still has pending/running/queued job
    statement = select(Job).join(Workflow).where(
        and_(
            Workflow.user_id == UUID(user_id),
            Job.status.in_(["PENDING", "RUNNING", "QUEUED"])
        )
    )
    result = await session.exec(statement)
    left = result.first()
    
    if left is None:
        await r.srem("active_users", user_id)
        await r.delete(f"user_activity:{user_id}")  # Clear activity time record
        print(f"👋  {user_id} Release execution permission (All tasks completed))")
        
        # Release waiting queue head user
        next_uid = await r.lpop("waiting_users")
        if next_uid:
            await r.sadd("active_users", next_uid)
            await update_user_activity(r, next_uid)  # Record new user activity time
            print(f"✅ Waiting user {next_uid} activated")

# --- Helper function:Check and release timeout users ---
async def check_and_release_timeout_users(session, r):
    """Check if active users timeout (minutes of inactivity), automatically release"""
    import time
    TIMEOUT_SECONDS = 600  # 10minutes
    
    active_users = await r.smembers("active_users")
    current_time = time.time()
    timeout_users = []
    
    for user_id in active_users:
        activity_key = f"user_activity:{user_id}"
        last_activity_str = await r.get(activity_key)
        
        if last_activity_str is None:
            # If no activity record,may be old user,release immediately
            timeout_users.append(user_id)
            print(f"⏰  {user_id} No activity record, automatically release")
        else:
            last_activity = float(last_activity_str)
            elapsed = current_time - last_activity
            
            if elapsed > TIMEOUT_SECONDS:
                # Check if user really has no tasks
                statement = select(Job).join(Workflow).where(
                    and_(
                        Workflow.user_id == UUID(user_id),
                        Job.status.in_(["PENDING", "RUNNING", "QUEUED"])
                    )
                )
                result = await session.exec(statement)
                has_active_jobs = result.first() is not None
                
                if not has_active_jobs:
                    timeout_users.append(user_id)
                    print(f"⏰  {user_id} timeout (minutes of inactivity), automatically release")
    
    # Release timeout users
    for user_id in timeout_users:
        await r.srem("active_users", user_id)
        await r.delete(f"user_activity:{user_id}")
        print(f"👋  {user_id} released (timeout)))")
        
        # Release waiting queue head user
        next_uid = await r.lpop("waiting_users")
        if next_uid:
            await r.sadd("active_users", next_uid)
            await update_user_activity(r, next_uid)
            print(f"✅ Waiting user {next_uid} activated (released (timeout)))")

async def run_scheduler():
    print("🚀 Final fixed version scheduler (dual lock mechanism + automatically release) started...", flush=True)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    r = redis.from_url(REDIS_URL, decode_responses=True)
    
    # Timeout check counter(Check every seconds)
    timeout_check_counter = 0
    
    while True:
        try:
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            async with async_session() as session:
                
                # Check timeout users every seconds(approximately every loops,Assuming each loop takes seconds)
                timeout_check_counter += 1
                if timeout_check_counter >= 30:
                    timeout_check_counter = 0
                    await check_and_release_timeout_users(session, r)
                    # ✅ Fix: Also check if active users still have pending tasks
                    active_users = await r.smembers("active_users")
                    for user_id in active_users:
                        try:
                            statement = select(Job).join(Workflow).where(
                                and_(
                                    Workflow.user_id == UUID(user_id),
                                    Job.status.in_(["PENDING", "RUNNING", "QUEUED"])
                                )
                            )
                            result = await session.exec(statement)
                            has_active_jobs = result.first() is not None
                            if not has_active_jobs:
                                await release_user_if_done(session, r, user_id)
                        except Exception as e:
                            print(f"⚠️  Error checking user: {e}")
                
                # 1. Handle retry of failed tasks (Branch-Local)
                # Note:Actual task processing is done by worker.py,Here only handle retry logic
                # ✅ Fix: Exclude CANCELLED status tasks
                statement_failed = select(Job).where(
                    and_(
                        Job.status == "FAILED",
                        Job.status != "CANCELLED"  # Exclude cancelled tasks
                    )
                )
                result_failed = await session.exec(statement_failed)
                failed_jobs = result_failed.all()
                
                for job in failed_jobs:
                    await retry_failed_job(session, job)

                # 2. Before scheduling new tasks,first check global worker count limit
                # Check tasks with RUNNING and QUEUED status(all occupy worker resources)
                # ⚠️ Important: Exclude CANCELLED status jobs
                statement_running = select(Job).where(
                    and_(
                        or_(Job.status == "RUNNING", Job.status == "QUEUED"),
                        Job.status != "CANCELLED"  # Explicitly exclude cancelled tasks
                    )
                )
                result_running = await session.exec(statement_running)
                running_jobs = result_running.all()
                running_count = len(running_jobs)
                
                # ✅ Debug information: Show currently running tasks and branches
                if running_count > 0:
                    branch_info = {}
                    for j in running_jobs:
                        branch = j.branch_id
                        if branch not in branch_info:
                            branch_info[branch] = {"RUNNING": 0, "QUEUED": 0}
                        branch_info[branch][j.status] = branch_info[branch].get(j.status, 0) + 1
                    print(f"   📊 Current active tasks ({running_count}/{MAX_WORKERS}): {branch_info}")
                free_slots = MAX_WORKERS - running_count
                
                # Debug: Show which branches are currently active
                active_branches = {}
                for j in running_jobs:
                    if j.branch_id not in active_branches:
                        active_branches[j.branch_id] = []
                    active_branches[j.branch_id].append(f"{j.status}")
                if active_branches:
                    branch_info = ", ".join([f"{bid}: {', '.join(stats)}" for bid, stats in active_branches.items()])
                    print(f"      📊 Active branches: {branch_info}")
                
                if free_slots <= 0:
                    print(f"      ⚠️ Global workers full ({running_count}/{MAX_WORKERS}),Skip this scheduling")
                    await asyncio.sleep(2)
                    continue
                
                print(f"      📊 Global worker status: {running_count}/{MAX_WORKERS} running, {free_slots} available slots")
                
                # =================================================
                # 🎯 Global FIFO scheduling:Sort across workflows by created_at
                # =================================================
                # Find all PENDING jobs,Sort by created_at (FIFO))
                # ✅ Fix: Directly query PENDING status(PENDING status cannot be CANCELLED at the same time)
                # But for safety,we filter again in memory
                statement_pending = select(Job).where(
                    Job.status == "PENDING"
                ).order_by(Job.created_at)
                result_pending = await session.exec(statement_pending)
                pending_jobs = result_pending.all()
                
                # ✅ Additional filtering:Ensure no CANCELLED status tasks(double insurance)
                # Refresh objects to ensure getting latest status
                for job in pending_jobs:
                    await session.refresh(job)
                pending_jobs = [j for j in pending_jobs if j.status == "PENDING"]
                
                if pending_jobs:
                    print(f"🔍 Scanned pending tasks(Global FIFO sorting)", flush=True)
                
                if not pending_jobs:
                    # No pending tasks,Check if any workflow needs to be marked as completed
                    statement_wf = select(Workflow).where(Workflow.status == "PENDING").options(selectinload(Workflow.jobs))
                    result_wf = await session.exec(statement_wf)
                    workflows = result_wf.all()
                    for wf in workflows:
                        all_succeeded = all(j.status == "SUCCEEDED" for j in wf.jobs)
                        if all_succeeded:
                            print(f"🎉 Workflow {wf.id} all completed！")
                            wf.status = "COMPLETED"
                            session.add(wf)
                            await session.commit()
                            # Release user and activate next waiting user
                            await release_user_if_done(session, r, str(wf.user_id))
                    await asyncio.sleep(2)
                    continue
                
                print(f"🔍 Scanned pending tasks(Global FIFO sorting)")
                
                # memory lock and counter
                active_branches_in_loop = set()
                started_this_tick = 0
                
                # Preload all related workflows and jobs,for dependency checking
                # Collect all needed workflow_ids
                workflow_ids = list({job.workflow_id for job in pending_jobs})
                workflows_dict = {}
                if workflow_ids:
                    # Query workflows one by one(Although slightly less efficient,but syntax is clearer)
                    for wf_id in workflow_ids:
                        statement_wf = select(Workflow).where(Workflow.id == wf_id).options(selectinload(Workflow.jobs))
                        result_wf = await session.exec(statement_wf)
                        wf = result_wf.first()
                        if wf:
                            workflows_dict[wf.id] = wf
                
                for job in pending_jobs:
                    # check 4: Global worker count limit(Check first,avoid unnecessary queries)
                    if started_this_tick >= free_slots:
                        print(f"      ⚠️  ({free_slots}),")
                        break
                    
                    #  job  workflow
                    wf = workflows_dict.get(job.workflow_id)
                    if not wf:
                        continue
                    
                    user_id = str(wf.user_id)
                    
                    # (queued)
                    ok = await ensure_user_active(r, user_id)
                    if not ok:
                        # ✅ Debug information:Gained execution permission
                        active_count = await r.scard("active_users")
                        waiting_count = await r.llen("waiting_users")
                        print(f"      ⏸️   {user_id[:8]}... Gained execution permission (: {active_count}/{MAX_ACTIVE_USERS}, : {waiting_count})")
                        continue
                    
                    # check 1: 
                    completed_job_ids = {str(j.id) for j in wf.jobs if j.status == "SUCCEEDED"}
                    parent_ids = json.loads(job.parent_ids_json)
                    parents_done = all(pid in completed_job_ids for pid in parent_ids)
                    if not parents_done:
                        continue 
                    
                    # check 2: memory lock (Check if someone preempted in this loop)
                    # Note:memory lockonly prevent duplicate start of same branch tasks within same loop
                    if job.branch_id in active_branches_in_loop:
                        print(f"      ⏳ branch {job.branch_id} busy (memory lock,started),Job {job.name} queued...")
                        continue

                    # check 3: database lock (Check if there are RUNNING tasks,globally across workflows)
                    # Key:only check RUNNING status,do not check QUEUED status
                    # :
                    # - RUNNING :task is running,other tasks in same branch must wait(serial execution)
                    # - QUEUED :task is queued but not running,worker will be processed in FIFO order
                    #   QUEUED tasks from different branches can run in parallel(controlled by worker count limit)
                    #   Multiple QUEUED tasks in same branch will be processed sequentially by workers(FIFO)
                    branch_busy_db = await is_branch_busy(session, job.branch_id)
                    if branch_busy_db:
                        # If database is busy(has RUNNING tasks),memory lock,prevent subsequent tasks from trying
                        active_branches_in_loop.add(job.branch_id) 
                        print(f"      ⏳ branch {job.branch_id} busy (DB,has RUNNING tasks),Job {job.name} queued...")
                        continue
                    
                    # Debug: Log branch check result
                    print(f"      ✅ branch {job.branch_id} idle(no RUNNING tasks),Job {job.name} can start")
                    
                    # --- , ---
                    print(f"   ▶️ Start Job: {job.name} (Branch: {job.branch_id}, Workflow: {wf.id})")
                    print(f"      📊 Current global state: {running_count}/{MAX_WORKERS} workers, {free_slots} ")
                    
                    job.status = "QUEUED"  # Change to QUEUED,wait for worker to process
                    session.add(job)
                    await session.commit()
                    
                    # Update user activity time(when task starts)
                    await update_user_activity(r, user_id)
                    
                    # 🔐 Key:Lock immediately！(prevent duplicate start of same branch within same loop)
                    # Note:This lock is only valid in current loop,will not prevent tasks from different branches from running in parallel
                    active_branches_in_loop.add(job.branch_id)
                    started_this_tick += 1
                    
                    # Debug: Show which branches are now active
                    print(f"      🔒 branch: {job.branch_id} (memory lock)")
                
                # Check if all workflows are completed
                for wf in workflows_dict.values():
                    all_succeeded = all(j.status == "SUCCEEDED" for j in wf.jobs)
                    if all_succeeded and wf.status == "PENDING":
                        print(f"🎉 Workflow {wf.id} all completed！")
                        wf.status = "COMPLETED"
                        session.add(wf)
                        await session.commit()
                        # Release user and activate next waiting user
                        await release_user_if_done(session, r, str(wf.user_id))

            await asyncio.sleep(2)

        except Exception as e:
            print(f"💥 Error: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(run_scheduler())
