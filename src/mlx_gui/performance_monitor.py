"""
Performance monitoring utilities for MLX-GUI.
Provides real-time performance metrics and analysis tools.
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import psutil
import gc

from .database import get_database_manager
from .model_manager import get_model_manager
from .inference_queue_manager import get_inference_manager
from .system_monitor import get_system_monitor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    memory_used_gb: float
    loaded_models_count: int
    total_model_memory_gb: float
    queue_size: int
    active_requests: int
    database_connections: int
    database_queries_per_second: float
    cache_hit_rate: float
    response_time_ms: float


@dataclass
class PerformanceAlert:
    """Performance alert."""
    timestamp: datetime
    level: str  # "warning", "error", "critical"
    message: str
    metric: str
    value: float
    threshold: float


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "response_time_ms": 5000.0,
            "queue_size": 10,
            "cache_hit_rate": 0.5,  # Below 50% is concerning
        }
        
        self._metrics_history: List[PerformanceMetrics] = []
        self._alerts_history: List[PerformanceAlert] = []
        self._max_history_size = 1000
        self._lock = threading.RLock()
        
        # Performance tracking
        self._last_metrics_time = time.time()
        self._query_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._response_times: List[float] = []
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 5.0  # 5 seconds
    
    def start_monitoring(self):
        """Start continuous performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="performance_monitor",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                metrics = self.collect_metrics()
                self._metrics_history.append(metrics)
                
                # Keep history size manageable
                if len(self._metrics_history) > self._max_history_size:
                    self._metrics_history.pop(0)
                
                # Check for alerts
                alerts = self.check_alerts(metrics)
                for alert in alerts:
                    self._alerts_history.append(alert)
                    logger.warning(f"Performance alert: {alert.message}")
                
                # Keep alerts history manageable
                if len(self._alerts_history) > self._max_history_size:
                    self._alerts_history.pop(0)
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(self._monitor_interval)
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        # Application metrics
        model_manager = get_model_manager()
        inference_manager = get_inference_manager()
        db_manager = get_database_manager()
        
        system_status = model_manager.get_system_status()
        
        # Calculate cache hit rate
        total_cache_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = self._cache_hits / total_cache_requests if total_cache_requests > 0 else 1.0
        
        # Calculate average response time
        avg_response_time = sum(self._response_times) / len(self._response_times) if self._response_times else 0.0
        
        # Calculate queries per second
        current_time = time.time()
        time_diff = current_time - self._last_metrics_time
        queries_per_second = self._query_count / time_diff if time_diff > 0 else 0.0
        
        # Reset counters
        self._last_metrics_time = current_time
        self._query_count = 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            memory_used_gb=memory.used / (1024**3),
            loaded_models_count=system_status["models"]["loaded_count"],
            total_model_memory_gb=system_status["models"]["total_memory_gb"],
            queue_size=system_status["models"]["queue_size"],
            active_requests=0,  # Will be calculated per model
            database_connections=0,  # Will be calculated from pool
            database_queries_per_second=queries_per_second,
            cache_hit_rate=cache_hit_rate,
            response_time_ms=avg_response_time
        )
        
        return metrics
    
    def check_alerts(self, metrics: PerformanceMetrics) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts."""
        alerts = []
        
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="warning" if metrics.cpu_percent < 90 else "critical",
                message=f"High CPU usage: {metrics.cpu_percent:.1f}%",
                metric="cpu_percent",
                value=metrics.cpu_percent,
                threshold=self.alert_thresholds["cpu_percent"]
            ))
        
        if metrics.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="warning" if metrics.memory_percent < 95 else "critical",
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                metric="memory_percent",
                value=metrics.memory_percent,
                threshold=self.alert_thresholds["memory_percent"]
            ))
        
        if metrics.queue_size > self.alert_thresholds["queue_size"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="warning",
                message=f"Large request queue: {metrics.queue_size} requests",
                metric="queue_size",
                value=metrics.queue_size,
                threshold=self.alert_thresholds["queue_size"]
            ))
        
        if metrics.cache_hit_rate < self.alert_thresholds["cache_hit_rate"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="warning",
                message=f"Low cache hit rate: {metrics.cache_hit_rate:.1%}",
                metric="cache_hit_rate",
                value=metrics.cache_hit_rate,
                threshold=self.alert_thresholds["cache_hit_rate"]
            ))
        
        if metrics.response_time_ms > self.alert_thresholds["response_time_ms"]:
            alerts.append(PerformanceAlert(
                timestamp=metrics.timestamp,
                level="warning" if metrics.response_time_ms < 10000 else "critical",
                message=f"Slow response time: {metrics.response_time_ms:.0f}ms",
                metric="response_time_ms",
                value=metrics.response_time_ms,
                threshold=self.alert_thresholds["response_time_ms"]
            ))
        
        return alerts
    
    def record_query(self):
        """Record a database query for metrics."""
        self._query_count += 1
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self._cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self._cache_misses += 1
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time measurement."""
        self._response_times.append(response_time_ms)
        
        # Keep only last 100 measurements
        if len(self._response_times) > 100:
            self._response_times.pop(0)
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.collect_metrics()
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                metrics for metrics in self._metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def get_alerts_history(self, minutes: int = 60) -> List[PerformanceAlert]:
        """Get alerts history for the last N minutes."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                alert for alert in self._alerts_history
                if alert.timestamp >= cutoff_time
            ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        current_metrics = self.get_current_metrics()
        recent_metrics = self.get_metrics_history(minutes=10)
        recent_alerts = self.get_alerts_history(minutes=10)
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Get system info
        system_monitor = get_system_monitor()
        system_info = system_monitor.get_system_info()
        
        return {
            "current": asdict(current_metrics),
            "averages_10min": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "response_time_ms": avg_response_time,
            },
            "system_info": {
                "total_memory_gb": system_info.memory.total_gb,
                "available_memory_gb": system_info.memory.available_gb,
                "platform": system_info.platform,
                "architecture": system_info.architecture,
            },
            "alerts": {
                "recent_count": len(recent_alerts),
                "critical_count": len([a for a in recent_alerts if a.level == "critical"]),
                "warning_count": len([a for a in recent_alerts if a.level == "warning"]),
            },
            "performance_score": self._calculate_performance_score(current_metrics),
        }
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate a performance score from 0-100."""
        score = 100.0
        
        # CPU penalty
        if metrics.cpu_percent > 80:
            score -= (metrics.cpu_percent - 80) * 2
        
        # Memory penalty
        if metrics.memory_percent > 80:
            score -= (metrics.memory_percent - 80) * 2
        
        # Response time penalty
        if metrics.response_time_ms > 1000:
            score -= (metrics.response_time_ms - 1000) / 100
        
        # Queue penalty
        if metrics.queue_size > 5:
            score -= (metrics.queue_size - 5) * 5
        
        # Cache hit rate penalty
        if metrics.cache_hit_rate < 0.8:
            score -= (0.8 - metrics.cache_hit_rate) * 50
        
        return max(0.0, min(100.0, score))
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Suggest performance optimizations."""
        current_metrics = self.get_current_metrics()
        suggestions = []
        
        if current_metrics.cpu_percent > 70:
            suggestions.append({
                "issue": "High CPU usage",
                "suggestion": "Consider reducing concurrent model loading or increasing model inactivity timeout",
                "priority": "high"
            })
        
        if current_metrics.memory_percent > 80:
            suggestions.append({
                "issue": "High memory usage",
                "suggestion": "Unload unused models or reduce max_concurrent_models setting",
                "priority": "high"
            })
        
        if current_metrics.queue_size > 5:
            suggestions.append({
                "issue": "Large request queue",
                "suggestion": "Increase max_concurrent_requests_per_model or add more processing workers",
                "priority": "medium"
            })
        
        if current_metrics.cache_hit_rate < 0.7:
            suggestions.append({
                "issue": "Low cache hit rate",
                "suggestion": "Increase cache TTL values or add more caching layers",
                "priority": "medium"
            })
        
        if current_metrics.response_time_ms > 2000:
            suggestions.append({
                "issue": "Slow response times",
                "suggestion": "Check database performance, optimize queries, or increase connection pool size",
                "priority": "high"
            })
        
        return {
            "suggestions": suggestions,
            "performance_score": self._calculate_performance_score(current_metrics),
            "current_metrics": asdict(current_metrics)
        }


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None
_performance_monitor_lock = threading.Lock()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        with _performance_monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    global _performance_monitor
    if _performance_monitor:
        _performance_monitor.stop_monitoring()


def get_performance_summary() -> Dict[str, Any]:
    """Get a quick performance summary."""
    monitor = get_performance_monitor()
    return monitor.get_performance_summary()


def get_optimization_suggestions() -> Dict[str, Any]:
    """Get performance optimization suggestions."""
    monitor = get_performance_monitor()
    return monitor.optimize_performance() 