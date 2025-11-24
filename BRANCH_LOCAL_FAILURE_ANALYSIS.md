# Branch-Local Failure Analysis

## ✅ 功能已实现

### 1. 重试机制实现

**位置**: `app/scheduler.py:77-90`
```python
async def retry_failed_job(session, job: Job):
    """
    Call this function for retry when worker fails
    Branch-Local: Failure only makes subsequent tasks in same branch wait,
                  does not affect other branches
    """
    if job.status == "FAILED" and job.attempt < job.max_retries:
        job.attempt += 1
        job.status = "PENDING"  # 重置为 PENDING，等待重新调度
        job.error = None
        session.add(job)
        await session.commit()
        print(f"🔁 Job {job.name} retry {job.attempt}/{job.max_retries} (Branch: {job.branch_id})")
        return True
    return False
```

**调度器调用**: `app/scheduler.py:204-217`
- 每个调度周期检查所有 `FAILED` 状态的 jobs
- 如果 `attempt < max_retries`，自动重试

### 2. Worker 失败处理

**位置**: `app/worker.py:838-852`
```python
except Exception as e:
    print(f"💥 Processing failed: {e}")
    # 设置状态为 FAILED
    failed_job = await safe_update_job(session, job_id, {
        "status": "FAILED",
        "error": str(e)[:500]
    })
```

### 3. Branch-Local 隔离验证

**关键函数**: `is_branch_busy()` - `app/scheduler.py:26-41`
```python
async def is_branch_busy(session, branch_id):
    """
    Check if this branch has running tasks(global lock, across workflows)
    Note: only check RUNNING status, do not check QUEUED status
    Only RUNNING tasks will block other tasks in the same branch(serial execution)
    """
    statement = select(Job).where(
        Job.branch_id == branch_id,
        Job.status == "RUNNING"  # ⚠️ 只检查 RUNNING，不检查 FAILED
    )
```

**关键点**:
- ✅ `is_branch_busy()` **只检查 `RUNNING` 状态**
- ✅ **不检查 `FAILED` 状态**
- ✅ 这意味着：一个 branch 中的 job 失败后，**不会阻塞其他 branch**

## 📊 工作流程

### 场景：Branch A 中的 Job 失败

1. **Job A1 失败**:
   - Worker 设置状态: `RUNNING → FAILED`
   - 错误信息保存到 `job.error`

2. **调度器检测失败**:
   - 下一个调度周期（2秒后）
   - 发现 `FAILED` 状态且 `attempt < max_retries`
   - 调用 `retry_failed_job()`
   - 状态重置: `FAILED → PENDING`

3. **重试调度**:
   - `PENDING` job 等待依赖满足
   - 如果 branch A 没有其他 `RUNNING` job，可以重新开始
   - 状态变为: `PENDING → QUEUED → RUNNING`

4. **其他 Branch 不受影响**:
   - `is_branch_busy()` 只检查 `RUNNING` 状态
   - Branch B 的 jobs 可以继续运行
   - 只有 Branch A 中的后续 jobs 会等待（因为需要串行执行）

### 场景：Branch A 和 Branch B 并行

```
时间线：
T1: Branch A - Job1 (RUNNING) ✅
    Branch B - Job1 (RUNNING) ✅  ← 并行执行

T2: Branch A - Job1 (FAILED) ❌
    Branch B - Job1 (RUNNING) ✅  ← Branch B 继续运行，不受影响

T3: Branch A - Job1 (PENDING) 🔁  ← 重试中
    Branch B - Job1 (RUNNING) ✅  ← Branch B 继续运行

T4: Branch A - Job1 (RUNNING) 🔁  ← 重试执行
    Branch B - Job1 (SUCCEEDED) ✅
    Branch B - Job2 (RUNNING) ✅  ← Branch B 的 Job2 可以开始
```

## ✅ 结论

**功能已完全实现**：

1. ✅ **重试机制**: 失败 jobs 自动重试（最多 2 次）
2. ✅ **Branch-Local**: 失败只影响同一 branch 的后续 jobs
3. ✅ **跨 Branch 隔离**: 一个 branch 的失败不影响其他 branch
4. ✅ **状态管理**: `FAILED → PENDING → QUEUED → RUNNING`

## 🔍 验证方法

可以通过以下方式验证：

1. **创建两个 branch 的 jobs**:
   - Branch A: Job1, Job2
   - Branch B: Job1, Job2

2. **模拟失败**:
   - 在 worker 中临时抛出异常
   - 观察 Branch A 的 Job1 失败后，Branch B 是否继续运行

3. **观察日志**:
   - 应该看到: `🔁 Job ... retry 1/2 (Branch: branch_A)`
   - Branch B 的 jobs 应该不受影响

## ⚠️ 注意事项

1. **重试次数**: `max_retries = 2`（默认值）
2. **重试间隔**: 调度器每 2 秒检查一次
3. **最终失败**: 如果重试 2 次后仍失败，job 保持 `FAILED` 状态
4. **依赖关系**: 如果失败的 job 是其他 job 的父节点，子 jobs 会一直等待

