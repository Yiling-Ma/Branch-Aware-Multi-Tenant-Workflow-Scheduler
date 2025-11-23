# My Scheduler - 任务调度系统

## 阶段一：Infrastructure & Data Core（基础设施与数据核心）

### 概述

阶段一成功实现了全栈基础设施容器化和核心数据模型设计，为后续的任务调度功能打下了坚实基础。

### Mental Model（心智模型）

以下图表展示了系统从用户请求到数据存储的完整流程：

```
[用户 (User)]
     ⬇️  (发送 HTTP 请求)

[FastAPI (你的代码)]
     ⬇️  (解析 JSON，验证数据)

[SQLModel (ORM)]
     ⬇️  (转换成 SQL 语句)

[PostgreSQL (数据库)]
     ⬇️

📦 存储结果:

   User表: {id: "...", username: "test_user_1"}

   Workflow表: {id: "...", status: "PENDING"}

   Job表:
      - Job A (Tiling)
      - Job B (Branch 1) -> 依赖 Job A
      - Job C (Branch 2) -> 依赖 Job A
```

**流程说明：**

1. **用户层**: 用户通过 HTTP 请求（如 `POST /workflows/`）提交任务工作流
2. **API 层**: FastAPI 接收请求，使用 Pydantic 验证 JSON 数据格式
3. **ORM 层**: SQLModel 将 Python 对象转换为 SQL 语句
4. **数据库层**: PostgreSQL 执行 SQL，持久化存储数据

**数据存储示例：**

- **User 表**: 存储用户基本信息
- **Workflow 表**: 存储工作流元数据（状态、创建时间等）
- **Job 表**: 存储具体任务，包括：
  - 任务类型（如 Tiling、Inference）
  - 分支标识（Branch 1、Branch 2）
  - 依赖关系（通过 `parent_ids_json` 存储）

### 核心成果

#### 1. 全栈基础设施容器化

使用 `docker-compose` 成功部署了现代分布式系统的标准组件：

- **PostgreSQL 13**: 主数据库，用于持久化存储
- **Redis 6**: 缓存和消息队列服务
- **FastAPI 应用**: 高性能异步 Web 框架

#### 2. 核心数据模型（Database Schema）

设计并部署了三张关键表：

- **User（用户表）**: 存储用户信息
- **Workflow（工作流表）**: 表示一个完整的工作流，包含多个任务
- **Job（任务表）**: 存储具体的任务信息

**核心设计特性：**

- **DAG（有向无环图）存储结构**: 通过 `parent_ids_json` 字段，数据库能够记录任务之间的依赖关系（如"任务 B 必须在任务 A 之后执行"）
- **Branch-Aware（分支感知）**: 通过 `branch_id` 字段，为调度器识别"哪些任务可以并行执行"提供数据基础

#### 3. RESTful API 接口

实现了以下 API 端点：

- `POST /users/`: 创建新用户
- `POST /workflows/`: 提交包含依赖关系的任务工作流

**技术亮点：**

- 使用 "Code-First"（代码优先）开发模式
- 自动处理任务依赖关系的解析和存储
- 支持通过索引引用方式定义任务依赖

#### 4. 技术栈

- **FastAPI**: 高性能异步 Web 框架
- **SQLModel**: 结合 SQLAlchemy 和 Pydantic 的现代 ORM
- **Asyncpg**: 异步 PostgreSQL 驱动，支持高并发
- **Docker Compose**: 容器编排工具

### 关键技术决策

1. **JSON 存储依赖关系**: 将父任务 ID 列表存储为 JSON 字符串格式，在关系型数据库中实现轻量级的图结构存储
2. **异步编程**: 全面使用 `async/await`，确保高并发性能
3. **自动 API 文档**: 利用 FastAPI 自带的 Swagger UI，实现可视化 API 测试

---

## 快速开始

### 前置要求

- Docker 和 Docker Compose
- Python 3.10+
- pip 或 conda

### 安装步骤

1. **克隆项目并进入目录**
   ```bash
   cd my-scheduler
   ```

2. **安装 Python 依赖**
   ```bash
   # 使用 conda 环境（推荐）
   conda activate penn
   pip install -r requirements.txt
   
   # 或使用 pip
   pip install -r requirements.txt
   ```

3. **启动数据库和 Redis 服务**
   ```bash
   docker-compose up -d db redis
   ```

4. **启动 FastAPI 应用**
   ```bash
   uvicorn app.main:app --reload
   ```

应用将在 `http://127.0.0.1:8000` 启动。

### 验证服务状态

检查 Docker 服务是否正常运行：

```bash
docker-compose ps
```

应该看到 `db` 和 `redis` 服务状态为 `healthy`。

---

## 测试阶段一功能

### 方法一：使用 Swagger UI（推荐）

1. 在浏览器中打开：http://127.0.0.1:8000/docs
2. 在 Swagger UI 中可以：
   - 查看所有 API 端点
   - 测试 API 接口
   - 查看请求/响应格式

### 方法二：使用 curl 命令

#### 1. 创建用户

```bash
curl -X POST "http://127.0.0.1:8000/users/" \
  -H "Content-Type: application/json" \
  -d '{"username": "test_user"}'
```

**响应示例：**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "test_user",
  "created_at": "2025-11-23T15:10:28.976Z"
}
```

**保存返回的 `id`，后续创建 Workflow 时需要用到。**

#### 2. 创建 Workflow（包含任务依赖）

```bash
curl -X POST "http://127.0.0.1:8000/workflows/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "jobs": [
      {
        "name": "Job1: 切片任务",
        "job_type": "tiling",
        "branch_id": "main",
        "parent_indices": []
      },
      {
        "name": "Job2: 推理任务",
        "job_type": "inference",
        "branch_id": "main",
        "parent_indices": [0]
      },
      {
        "name": "Job3: 后处理任务",
        "job_type": "postprocessing",
        "branch_id": "main",
        "parent_indices": [1]
      }
    ]
  }'
```

**说明：**
- `parent_indices`: 数组中的索引，指向 `jobs` 列表中前面的任务
- 例如 `[0]` 表示依赖第一个任务（Job1）
- 例如 `[1]` 表示依赖第二个任务（Job2）

**响应示例：**
```json
{
  "workflow_id": "fbb11f2e-f5a1-4d13-855d-30a9ab5e1a22",
  "job_count": 3
}
```


---

## 项目结构

```
my-scheduler/
├── app/
│   ├── main.py              # FastAPI 应用入口和 API 路由
│   ├── models.py            # 数据模型（User, Workflow, Job）
│   ├── db/
│   │   └── __init__.py      # 数据库连接和初始化
│   └── core/
│       └── config.py         # 配置管理
├── docker-compose.yml        # Docker 服务编排
├── Dockerfile               # 应用镜像构建
├── requirements.txt         # Python 依赖
└── README.md                # 本文档
```

---

## 常见问题

### 1. 数据库连接失败

**错误**: `Connect call failed ('127.0.0.1', 5432)`

**解决**: 确保 Docker 服务已启动
```bash
docker-compose up -d db redis
docker-compose ps  # 验证服务状态
```

### 2. 表结构不匹配

**错误**: `column "job_type" does not exist`

**解决**: 重启应用，`init_db()` 会自动重新创建表结构
```bash
# 停止应用（Ctrl+C），然后重新启动
uvicorn app.main:app --reload
```

### 3. 端口被占用

**错误**: `Address already in use`

**解决**: 更改端口或停止占用端口的进程
```bash
uvicorn app.main:app --reload --port 8001
```

---

## 下一步

阶段一完成后，可以继续实现：

- **阶段二**: 任务调度器（Scheduler）
- **阶段三**: Worker 执行器
- **阶段四**: 任务状态管理和监控

---

## 许可证

[根据项目需要添加许可证信息]

