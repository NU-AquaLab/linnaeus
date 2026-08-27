"""
Performance monitoring and optimization for two-stage pipeline.

This module provides comprehensive monitoring, profiling, and optimization
capabilities for the two-stage hierarchical classification system.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for pipeline components."""

    component_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    cpu_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = time.time()

        if self.duration is None:
            self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_name": self.component_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "memory_peak": self.memory_peak,
            "cpu_usage": self.cpu_usage,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring for two-stage pipeline.

    Tracks timing, memory usage, CPU utilization, and other performance
    metrics across all pipeline components.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance monitor.

        Parameters
        ----------
        max_history : int
            Maximum number of metrics to keep in history.
        """
        self.max_history = max_history
        self.metrics_history: Deque[PerformanceMetrics] = deque(maxlen=max_history)
        self.component_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.active_metrics: Dict[str, PerformanceMetrics] = {}
        self.system_monitor_active = False
        self.system_metrics: Deque[Dict[str, Any]] = deque(maxlen=max_history)
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        logger.info(f"Initialized PerformanceMonitor with max_history={max_history}")

    def start_system_monitoring(self, interval: float = 1.0):
        """
        Start system-wide monitoring in background thread.

        Parameters
        ----------
        interval : float
            Monitoring interval in seconds.
        """
        if self.system_monitor_active:
            logger.warning("System monitoring already active")
            return

        self.system_monitor_active = True
        self._stop_monitoring.clear()

        def monitor_system():
            """Background system monitoring."""
            while not self._stop_monitoring.is_set():
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage("/")

                    # Network I/O (if available)
                    try:
                        net_io = psutil.net_io_counters()
                        network_metrics = {
                            "bytes_sent": net_io.bytes_sent,
                            "bytes_recv": net_io.bytes_recv,
                            "packets_sent": net_io.packets_sent,
                            "packets_recv": net_io.packets_recv,
                        }
                    except:
                        network_metrics = {}

                    system_metric = {
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available": memory.available,
                        "memory_used": memory.used,
                        "disk_percent": disk.percent,
                        "disk_free": disk.free,
                        **network_metrics,
                    }

                    self.system_metrics.append(system_metric)

                except Exception as e:
                    logger.warning(f"System monitoring error: {e}")

                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        self._monitor_thread.start()

        logger.info(f"Started system monitoring with {interval}s interval")

    def stop_system_monitoring(self):
        """Stop system monitoring."""
        if not self.system_monitor_active:
            return

        self._stop_monitoring.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        self.system_monitor_active = False
        logger.info("Stopped system monitoring")

    @contextmanager
    def monitor_component(self, component_name: str, **metadata):
        """
        Context manager for monitoring component performance.

        Parameters
        ----------
        component_name : str
            Name of the component being monitored.
        **metadata
            Additional metadata to track.

        Examples
        --------
        >>> monitor = PerformanceMonitor()
        >>> with monitor.monitor_component("stage1_llm", batch_size=10):
        ...     # Component execution here
        ...     result = component.process()
        """
        # Start monitoring
        metrics = PerformanceMetrics(
            component_name=component_name,
            start_time=time.time(),
            memory_before=self._get_memory_usage(),
            metadata=metadata,
        )

        self.active_metrics[component_name] = metrics

        try:
            yield metrics
            metrics.success = True

        except Exception as e:
            metrics.success = False
            metrics.error_message = str(e)
            logger.error(f"Component {component_name} failed: {e}")
            raise

        finally:
            # Finalize metrics
            metrics.end_time = time.time()
            metrics.memory_after = self._get_memory_usage()
            metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
            metrics.finalize()

            # Store metrics
            self.metrics_history.append(metrics)
            self.component_stats[component_name].append(metrics)

            # Remove from active
            self.active_metrics.pop(component_name, None)

            logger.debug(
                f"Component {component_name} completed in {metrics.duration:.3f}s"
            )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0

    def get_component_summary(self, component_name: str) -> Dict[str, Any]:
        """
        Get performance summary for a specific component.

        Parameters
        ----------
        component_name : str
            Name of the component.

        Returns
        -------
        Dict[str, Any]
            Performance summary statistics.
        """
        component_metrics = self.component_stats.get(component_name, [])

        if not component_metrics:
            return {"error": f"No metrics found for component {component_name}"}

        durations = [m.duration for m in component_metrics if m.duration is not None]
        memory_usage = [
            m.memory_after - m.memory_before
            for m in component_metrics
            if m.memory_before is not None and m.memory_after is not None
        ]

        success_rate = sum(1 for m in component_metrics if m.success) / len(
            component_metrics
        )

        summary = {
            "component_name": component_name,
            "total_calls": len(component_metrics),
            "success_rate": success_rate,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "total_duration": sum(durations) if durations else 0,
        }

        if memory_usage:
            summary.update(
                {
                    "avg_memory_delta": sum(memory_usage) / len(memory_usage),
                    "max_memory_delta": max(memory_usage),
                    "min_memory_delta": min(memory_usage),
                }
            )

        return summary

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get overall pipeline performance summary.

        Returns
        -------
        Dict[str, Any]
            Overall performance statistics.
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}

        # Overall statistics
        total_calls = len(self.metrics_history)
        successful_calls = sum(1 for m in self.metrics_history if m.success)
        total_duration = sum(m.duration for m in self.metrics_history if m.duration)

        # Component breakdown
        component_summaries = {}
        for component_name in self.component_stats:
            component_summaries[component_name] = self.get_component_summary(
                component_name
            )

        # Recent performance (last 100 calls)
        recent_metrics = list(self.metrics_history)[-100:]
        recent_durations = [m.duration for m in recent_metrics if m.duration]
        recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(
            recent_metrics
        )

        summary = {
            "total_calls": total_calls,
            "success_rate": successful_calls / total_calls,
            "total_duration": total_duration,
            "avg_duration": total_duration / total_calls if total_calls > 0 else 0,
            "component_summaries": component_summaries,
            "recent_performance": {
                "calls": len(recent_metrics),
                "success_rate": recent_success_rate,
                "avg_duration": (
                    sum(recent_durations) / len(recent_durations)
                    if recent_durations
                    else 0
                ),
            },
        }

        return summary

    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """
        Get system metrics summary.

        Returns
        -------
        Dict[str, Any]
            System performance statistics.
        """
        if not self.system_metrics:
            return {"error": "No system metrics available"}

        recent_metrics = list(self.system_metrics)[-100:]  # Last 100 samples

        cpu_values = [m["cpu_percent"] for m in recent_metrics]
        memory_values = [m["memory_percent"] for m in recent_metrics]

        summary = {
            "samples": len(recent_metrics),
            "time_range": {
                "start": recent_metrics[0]["timestamp"],
                "end": recent_metrics[-1]["timestamp"],
                "duration": recent_metrics[-1]["timestamp"]
                - recent_metrics[0]["timestamp"],
            },
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
            },
        }

        return summary

    def export_metrics(self, output_path: str, include_system_metrics: bool = True):
        """
        Export all metrics to file.

        Parameters
        ----------
        output_path : str
            Path to save metrics.
        include_system_metrics : bool
            Whether to include system metrics.
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "pipeline_summary": self.get_pipeline_summary(),
            "component_metrics": [m.to_dict() for m in self.metrics_history],
        }

        if include_system_metrics and self.system_metrics:
            export_data["system_metrics"] = list(self.system_metrics)
            export_data["system_summary"] = self.get_system_metrics_summary()

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {output_path}")

    def reset_metrics(self):
        """Reset all stored metrics."""
        self.metrics_history.clear()
        self.component_stats.clear()
        self.active_metrics.clear()
        self.system_metrics.clear()
        logger.info("Reset all performance metrics")


class PerformanceOptimizer:
    """
    Performance optimization recommendations and automatic tuning.

    Analyzes performance metrics and suggests optimizations for the
    two-stage pipeline configuration.
    """

    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize performance optimizer.

        Parameters
        ----------
        monitor : PerformanceMonitor
            Performance monitor instance.
        """
        self.monitor = monitor
        logger.info("Initialized PerformanceOptimizer")

    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """
        Analyze performance bottlenecks in the pipeline.

        Returns
        -------
        Dict[str, Any]
            Bottleneck analysis and recommendations.
        """
        pipeline_summary = self.monitor.get_pipeline_summary()

        if "component_summaries" not in pipeline_summary:
            return {"error": "Insufficient data for bottleneck analysis"}

        component_summaries = pipeline_summary["component_summaries"]

        # Find slowest components
        slowest_components = sorted(
            [
                (name, stats["avg_duration"])
                for name, stats in component_summaries.items()
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        # Find components with low success rates
        unreliable_components = [
            (name, stats["success_rate"])
            for name, stats in component_summaries.items()
            if stats["success_rate"] < 0.95
        ]

        # Memory usage analysis
        memory_intensive = []
        for name, stats in component_summaries.items():
            if (
                "avg_memory_delta" in stats and stats["avg_memory_delta"] > 100
            ):  # > 100MB
                memory_intensive.append((name, stats["avg_memory_delta"]))

        analysis = {
            "slowest_components": slowest_components[:5],
            "unreliable_components": unreliable_components,
            "memory_intensive_components": sorted(
                memory_intensive, key=lambda x: x[1], reverse=True
            ),
            "total_pipeline_time": pipeline_summary.get("avg_duration", 0),
            "overall_success_rate": pipeline_summary.get("success_rate", 0),
        }

        return analysis

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate performance optimization recommendations.

        Returns
        -------
        List[Dict[str, Any]]
            List of optimization recommendations.
        """
        bottlenecks = self.analyze_bottlenecks()
        recommendations = []

        if "error" in bottlenecks:
            return [{"type": "error", "message": bottlenecks["error"]}]

        # Slow component recommendations
        if bottlenecks["slowest_components"]:
            slowest_name, slowest_time = bottlenecks["slowest_components"][0]

            if slowest_time > 5.0:  # More than 5 seconds
                recommendations.append(
                    {
                        "type": "performance",
                        "priority": "high",
                        "component": slowest_name,
                        "issue": f"Component takes {slowest_time:.2f}s on average",
                        "recommendations": [
                            "Consider increasing batch size for better throughput",
                            "Enable parallel processing if not already active",
                            "Check for I/O bottlenecks or API rate limits",
                            "Consider caching for repeated operations",
                        ],
                    }
                )

        # Reliability recommendations
        if bottlenecks["unreliable_components"]:
            for comp_name, success_rate in bottlenecks["unreliable_components"]:
                recommendations.append(
                    {
                        "type": "reliability",
                        "priority": "high",
                        "component": comp_name,
                        "issue": f"Success rate is {success_rate:.1%}",
                        "recommendations": [
                            "Increase retry attempts in error handling",
                            "Improve input validation",
                            "Add fallback mechanisms",
                            "Review error logs for common failure patterns",
                        ],
                    }
                )

        # Memory usage recommendations
        if bottlenecks["memory_intensive_components"]:
            for comp_name, memory_delta in bottlenecks["memory_intensive_components"]:
                if memory_delta > 500:  # > 500MB
                    recommendations.append(
                        {
                            "type": "memory",
                            "priority": "medium",
                            "component": comp_name,
                            "issue": f"Uses {memory_delta:.1f}MB on average",
                            "recommendations": [
                                "Process data in smaller chunks",
                                "Clear intermediate results when possible",
                                "Consider streaming processing for large datasets",
                                "Monitor for memory leaks",
                            ],
                        }
                    )

        # Overall performance recommendations
        total_time = bottlenecks["total_pipeline_time"]
        if total_time > 10.0:  # More than 10 seconds average
            recommendations.append(
                {
                    "type": "overall",
                    "priority": "medium",
                    "component": "pipeline",
                    "issue": f"Overall pipeline time is {total_time:.2f}s",
                    "recommendations": [
                        "Optimize batch size configuration",
                        "Enable parallel processing where possible",
                        "Consider pipeline-level caching",
                        "Review configuration for optimal settings",
                    ],
                }
            )

        # System-level recommendations
        system_summary = self.monitor.get_system_metrics_summary()
        if "cpu" in system_summary:
            avg_cpu = system_summary["cpu"]["avg"]
            if avg_cpu > 80:
                recommendations.append(
                    {
                        "type": "system",
                        "priority": "medium",
                        "component": "system",
                        "issue": f"High CPU usage: {avg_cpu:.1f}%",
                        "recommendations": [
                            "Consider scaling to multiple machines",
                            "Reduce concurrent operations",
                            "Optimize CPU-intensive operations",
                            "Monitor for CPU-bound bottlenecks",
                        ],
                    }
                )

        if not recommendations:
            recommendations.append(
                {
                    "type": "info",
                    "priority": "low",
                    "component": "pipeline",
                    "issue": "No significant performance issues detected",
                    "recommendations": [
                        "Continue monitoring performance",
                        "Consider load testing with larger datasets",
                        "Review configuration periodically",
                    ],
                }
            )

        return recommendations

    def suggest_batch_size(self, target_latency: float = 2.0) -> Dict[str, Any]:
        """
        Suggest optimal batch size based on performance data.

        Parameters
        ----------
        target_latency : float
            Target latency in seconds.

        Returns
        -------
        Dict[str, Any]
            Batch size recommendations.
        """
        # This would require batch size performance data
        # For now, provide general recommendations

        pipeline_summary = self.monitor.get_pipeline_summary()
        avg_duration = pipeline_summary.get("avg_duration", 0)

        if avg_duration == 0:
            return {"error": "No performance data available"}

        # Simple heuristic-based recommendations
        current_latency = avg_duration

        if current_latency > target_latency * 2:
            suggestion = {
                "recommendation": "increase",
                "current_latency": current_latency,
                "target_latency": target_latency,
                "suggested_change": "Increase batch size by 50-100%",
                "rationale": "High latency suggests batch size is too small",
            }
        elif current_latency < target_latency * 0.5:
            suggestion = {
                "recommendation": "decrease",
                "current_latency": current_latency,
                "target_latency": target_latency,
                "suggested_change": "Decrease batch size by 25-50%",
                "rationale": "Very low latency suggests over-batching",
            }
        else:
            suggestion = {
                "recommendation": "maintain",
                "current_latency": current_latency,
                "target_latency": target_latency,
                "suggested_change": "Current batch size appears optimal",
                "rationale": "Latency is within acceptable range",
            }

        return suggestion


class PerformanceProfiler:
    """
    Detailed performance profiling for pipeline components.

    Provides fine-grained timing and resource usage analysis.
    """

    def __init__(self):
        """Initialize performance profiler."""
        self.profiles: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        logger.info("Initialized PerformanceProfiler")

    @contextmanager
    def profile_function(self, function_name: str, **metadata):
        """
        Profile a specific function or code block.

        Parameters
        ----------
        function_name : str
            Name of the function being profiled.
        **metadata
            Additional metadata to track.
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss

        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss

            profile_data = {
                "timestamp": start_time,
                "duration": end_time - start_time,
                "memory_delta": (end_memory - start_memory) / 1024 / 1024,  # MB
                "success": success,
                "error": error,
                "metadata": metadata,
            }

            self.profiles[function_name].append(profile_data)

    def get_function_profile(self, function_name: str) -> Dict[str, Any]:
        """
        Get profile summary for a specific function.

        Parameters
        ----------
        function_name : str
            Name of the function.

        Returns
        -------
        Dict[str, Any]
            Profile summary.
        """
        profiles = self.profiles.get(function_name, [])

        if not profiles:
            return {"error": f"No profile data for function {function_name}"}

        durations = [p["duration"] for p in profiles]
        memory_deltas = [p["memory_delta"] for p in profiles]
        success_count = sum(1 for p in profiles if p["success"])

        return {
            "function_name": function_name,
            "call_count": len(profiles),
            "success_rate": success_count / len(profiles),
            "timing": {
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations),
            },
            "memory": {
                "avg_delta": sum(memory_deltas) / len(memory_deltas),
                "min_delta": min(memory_deltas),
                "max_delta": max(memory_deltas),
                "total_delta": sum(memory_deltas),
            },
        }

    def export_profiles(self, output_path: str):
        """
        Export all profiles to file.

        Parameters
        ----------
        output_path : str
            Path to save profiles.
        """
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "function_profiles": {
                func_name: self.get_function_profile(func_name)
                for func_name in self.profiles
            },
            "raw_profiles": dict(self.profiles),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported profiles to {output_path}")


# Export main classes
__all__ = [
    "PerformanceMetrics",
    "PerformanceMonitor",
    "PerformanceOptimizer",
    "PerformanceProfiler",
]
