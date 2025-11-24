"""
Prometheus metrics for observability
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import time
from typing import Dict, Optional

# --- Metrics Definitions ---

# Queue depth metrics (per branch and total)
queue_depth_total = Gauge(
    'scheduler_queue_depth_total',
    'Total number of jobs in queue (PENDING + QUEUED)',
    ['status']  # status: PENDING, QUEUED
)

queue_depth_by_branch = Gauge(
    'scheduler_queue_depth_by_branch',
    'Number of queued jobs per branch',
    ['branch_id', 'status']
)

# Active workers and jobs
worker_active_jobs = Gauge(
    'scheduler_worker_active_jobs',
    'Number of currently active jobs per worker',
    ['worker_id', 'job_type', 'status']  # status: RUNNING, QUEUED
)

worker_active_jobs_total = Gauge(
    'scheduler_worker_active_jobs_total',
    'Total number of active jobs across all workers',
    ['status']
)

# Job latency metrics
job_latency_seconds = Histogram(
    'scheduler_job_latency_seconds',
    'Job execution latency in seconds',
    ['job_type', 'status'],  # status: SUCCEEDED, FAILED
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]  # 1s to 1h
)

job_duration_seconds = Histogram(
    'scheduler_job_duration_seconds',
    'Total job duration from creation to completion',
    ['job_type', 'status'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
)

# Job status counters
jobs_total = Counter(
    'scheduler_jobs_total',
    'Total number of jobs processed',
    ['job_type', 'status']
)

# Workflow metrics
workflows_total = Counter(
    'scheduler_workflows_total',
    'Total number of workflows created',
    ['status']
)

# Rate limiting metrics
rate_limit_requests_total = Counter(
    'scheduler_rate_limit_requests_total',
    'Total number of rate limit requests',
    ['endpoint', 'result']  # result: allowed, denied
)

# Active users metric
active_users = Gauge(
    'scheduler_active_users',
    'Number of currently active users'
)

# Branch metrics
branch_jobs_running = Gauge(
    'scheduler_branch_jobs_running',
    'Number of running jobs per branch',
    ['branch_id']
)


class MetricsCollector:
    """Helper class to collect and update metrics"""
    
    @staticmethod
    def update_queue_depth(status_counts: Dict[str, int], branch_counts: Optional[Dict[str, Dict[str, int]]] = None):
        """
        Update queue depth metrics
        
        Args:
            status_counts: Dict with keys 'PENDING', 'QUEUED' and their counts
            branch_counts: Optional dict of {branch_id: {status: count}}
        """
        for status, count in status_counts.items():
            queue_depth_total.labels(status=status).set(count)
        
        if branch_counts:
            for branch_id, branch_status_counts in branch_counts.items():
                for status, count in branch_status_counts.items():
                    queue_depth_by_branch.labels(branch_id=branch_id, status=status).set(count)
    
    @staticmethod
    def update_active_jobs(worker_id: int, job_type: str, status: str, count: int = 1):
        """Update active jobs metric for a specific worker"""
        worker_active_jobs.labels(worker_id=worker_id, job_type=job_type, status=status).set(count)
    
    @staticmethod
    def update_active_jobs_total(status_counts: Dict[str, int]):
        """Update total active jobs across all workers"""
        for status, count in status_counts.items():
            worker_active_jobs_total.labels(status=status).set(count)
    
    @staticmethod
    def record_job_latency(job_type: str, status: str, latency_seconds: float):
        """Record job execution latency"""
        job_latency_seconds.labels(job_type=job_type, status=status).observe(latency_seconds)
    
    @staticmethod
    def record_job_duration(job_type: str, status: str, duration_seconds: float):
        """Record total job duration from creation to completion"""
        job_duration_seconds.labels(job_type=job_type, status=status).observe(duration_seconds)
    
    @staticmethod
    def increment_job_counter(job_type: str, status: str):
        """Increment job counter"""
        jobs_total.labels(job_type=job_type, status=status).inc()
    
    @staticmethod
    def increment_workflow_counter(status: str):
        """Increment workflow counter"""
        workflows_total.labels(status=status).inc()
    
    @staticmethod
    def update_active_users(count: int):
        """Update active users count"""
        active_users.set(count)
    
    @staticmethod
    def update_branch_jobs_running(branch_id: str, count: int):
        """Update running jobs count for a branch"""
        branch_jobs_running.labels(branch_id=branch_id).set(count)
    
    @staticmethod
    def record_rate_limit(endpoint: str, allowed: bool):
        """Record rate limit decision"""
        result = "allowed" if allowed else "denied"
        rate_limit_requests_total.labels(endpoint=endpoint, result=result).inc()


# Global metrics collector instance
metrics = MetricsCollector()


def get_metrics():
    """Get Prometheus metrics in text format"""
    return generate_latest(REGISTRY)


def get_metrics_content_type():
    """Get content type for Prometheus metrics"""
    return CONTENT_TYPE_LATEST

