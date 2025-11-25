from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
from app.db.session import init_db
from app.services.rate_limit import init_rate_limiter
from app.api.routes import router as api_router

app = FastAPI()

# Initialize rate limiter on startup
@app.on_event("startup")
async def startup_event():
    await init_rate_limiter()

# Initialize database tables on startup
@app.on_event("startup")
async def on_startup():
    await init_db()

# 1. Configure CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Mount static file directory (key step)
# This allows accessing /static/xxx.jpg to get files
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 3. Include API routes
app.include_router(api_router)
