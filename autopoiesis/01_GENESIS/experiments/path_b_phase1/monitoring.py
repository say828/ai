"""
GENESIS Phase 4: Real-time Monitoring Utilities

Tools for monitoring long-running experiments:
- Live progress tracking
- Resource usage monitoring
- Performance metrics
- Alert conditions
- Web dashboard (optional)

Usage:
    from monitoring import ExperimentMonitor

    monitor = ExperimentMonitor()
    monitor.start()

    for step in range(10000):
        stats = manager.step()
        monitor.update(step, stats)

    monitor.stop()
    monitor.print_report()
"""

import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import deque
import json
from pathlib import Path


class ExperimentMonitor:
    """
    Real-time experiment monitoring with alerts and reporting
    """

    def __init__(self, output_dir: Optional[str] = None,
                 alert_coherence_drop: float = 0.1,
                 alert_memory_percent: float = 90.0):
        """
        Initialize monitor

        Args:
            output_dir: Directory for logs (None = no file output)
            alert_coherence_drop: Alert if coherence drops by this amount
            alert_memory_percent: Alert if memory usage exceeds this percent
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.alert_coherence_drop = alert_coherence_drop
        self.alert_memory_percent = alert_memory_percent

        # Monitoring state
        self.start_time = None
        self.last_update_time = None
        self.is_running = False

        # Statistics tracking
        self.step_times = deque(maxlen=100)  # Last 100 step times
        self.coherence_history = deque(maxlen=1000)
        self.population_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=100)

        # Metrics
        self.total_steps = 0
        self.peak_memory_mb = 0
        self.alerts = []

        # Process info
        self.process = psutil.Process(os.getpid())

    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.is_running = True

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._log("Monitor started")

        print(f"[{self._timestamp()}] Monitor started")

    def update(self, step: int, stats: Dict):
        """
        Update monitor with new statistics

        Args:
            step: Current step number
            stats: Statistics dictionary
        """
        if not self.is_running:
            return

        current_time = time.time()

        # Step timing
        if self.last_update_time:
            step_time = current_time - self.last_update_time
            self.step_times.append(step_time)

        self.last_update_time = current_time
        self.total_steps = step

        # Extract metrics
        coherence = stats.get('avg_coherence', 0)
        population = stats.get('population_size', 0)

        self.coherence_history.append(coherence)
        self.population_history.append(population)

        # Resource usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        memory_percent = self.process.memory_percent()

        self.memory_history.append(memory_mb)
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

        # Check alerts
        self._check_alerts(coherence, memory_percent)

        # Log periodically
        if step % 100 == 0:
            self._log_status(step, stats, memory_mb, memory_percent)

    def stop(self):
        """Stop monitoring"""
        if not self.is_running:
            return

        self.is_running = False
        duration = time.time() - self.start_time

        print(f"\n[{self._timestamp()}] Monitor stopped")
        print(f"Total duration: {self._format_duration(duration)}")

        if self.output_dir:
            self._log("Monitor stopped")
            self._save_report()

    def _check_alerts(self, coherence: float, memory_percent: float):
        """Check for alert conditions"""
        # Coherence drop alert
        if len(self.coherence_history) > 10:
            recent_avg = sum(list(self.coherence_history)[-10:]) / 10
            if coherence < recent_avg - self.alert_coherence_drop:
                alert = f"⚠️  Coherence dropped: {coherence:.3f} (avg: {recent_avg:.3f})"
                self._alert(alert)

        # Memory alert
        if memory_percent > self.alert_memory_percent:
            alert = f"⚠️  High memory usage: {memory_percent:.1f}%"
            self._alert(alert)

        # Population collapse alert
        if len(self.population_history) > 10:
            recent_pop = list(self.population_history)[-10:]
            if all(p < recent_pop[0] * 0.5 for p in recent_pop[5:]):
                alert = f"⚠️  Population collapsing: {recent_pop[-1]} (was {recent_pop[0]})"
                self._alert(alert)

    def _alert(self, message: str):
        """Record and display alert"""
        timestamp = self._timestamp()
        alert_msg = f"[{timestamp}] {message}"

        self.alerts.append({'time': timestamp, 'message': message})
        print(f"\n{alert_msg}\n")

        if self.output_dir:
            with open(self.output_dir / 'alerts.log', 'a') as f:
                f.write(f"{alert_msg}\n")

    def _log_status(self, step: int, stats: Dict, memory_mb: float, memory_percent: float):
        """Log current status"""
        if not self.step_times:
            return

        avg_step_time = sum(self.step_times) / len(self.step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

        coherence = stats.get('avg_coherence', 0)
        population = stats.get('population_size', 0)

        elapsed = time.time() - self.start_time

        status = (
            f"[{self._timestamp()}] "
            f"Step {step:6,} | "
            f"Coherence: {coherence:.3f} | "
            f"Pop: {population:4d} | "
            f"Speed: {steps_per_sec:.2f} steps/s | "
            f"Memory: {memory_mb:.1f}MB ({memory_percent:.1f}%) | "
            f"Elapsed: {self._format_duration(elapsed)}"
        )

        print(status)

        if self.output_dir:
            self._log(status)

    def _log(self, message: str):
        """Write to log file"""
        if not self.output_dir:
            return

        with open(self.output_dir / 'monitor.log', 'a') as f:
            f.write(f"{message}\n")

    def _save_report(self):
        """Save monitoring report"""
        if not self.output_dir:
            return

        report = {
            'duration': time.time() - self.start_time,
            'total_steps': self.total_steps,
            'avg_step_time': sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            'peak_memory_mb': self.peak_memory_mb,
            'final_coherence': list(self.coherence_history)[-1] if self.coherence_history else 0,
            'final_population': list(self.population_history)[-1] if self.population_history else 0,
            'alerts': self.alerts
        }

        with open(self.output_dir / 'monitor_report.json', 'w') as f:
            json.dump(report, f, indent=2)

    def print_report(self):
        """Print monitoring report"""
        if not self.start_time:
            print("Monitor was not started")
            return

        print("\n" + "="*70)
        print("MONITORING REPORT")
        print("="*70)

        duration = time.time() - self.start_time

        print(f"\nDuration: {self._format_duration(duration)}")
        print(f"Total Steps: {self.total_steps:,}")

        if self.step_times:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            steps_per_sec = 1.0 / avg_step_time
            print(f"Avg Step Time: {avg_step_time:.4f}s")
            print(f"Throughput: {steps_per_sec:.2f} steps/sec")

        print(f"\nPeak Memory: {self.peak_memory_mb:.1f}MB")

        if self.coherence_history:
            initial = list(self.coherence_history)[0]
            final = list(self.coherence_history)[-1]
            print(f"\nCoherence: {initial:.3f} → {final:.3f} ({final-initial:+.3f})")

        if self.population_history:
            initial = list(self.population_history)[0]
            final = list(self.population_history)[-1]
            growth = (final / initial - 1) * 100 if initial > 0 else 0
            print(f"Population: {initial} → {final} ({growth:+.1f}%)")

        if self.alerts:
            print(f"\nAlerts: {len(self.alerts)}")
            for alert in self.alerts[-5:]:  # Show last 5
                print(f"  [{alert['time']}] {alert['message']}")

        print("="*70)

    def _timestamp(self) -> str:
        """Get formatted timestamp"""
        return datetime.now().strftime("%H:%M:%S")

    def _format_duration(self, seconds: float) -> str:
        """Format duration as human-readable string"""
        td = timedelta(seconds=int(seconds))
        hours = td.seconds // 3600
        minutes = (td.seconds % 3600) // 60
        secs = td.seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class ProgressEstimator:
    """
    Estimate time remaining for long experiments
    """

    def __init__(self, total_steps: int):
        """
        Initialize estimator

        Args:
            total_steps: Total number of steps
        """
        self.total_steps = total_steps
        self.start_time = None
        self.step_times = deque(maxlen=100)

    def update(self, current_step: int):
        """Update with current step"""
        if self.start_time is None:
            self.start_time = time.time()
            return

        current_time = time.time()
        elapsed = current_time - self.start_time

        # Estimate
        if current_step > 0:
            avg_time_per_step = elapsed / current_step
            remaining_steps = self.total_steps - current_step
            estimated_remaining = avg_time_per_step * remaining_steps

            percent = (current_step / self.total_steps) * 100

            return {
                'percent': percent,
                'elapsed': elapsed,
                'remaining': estimated_remaining,
                'eta': datetime.now() + timedelta(seconds=estimated_remaining),
                'speed': 1.0 / avg_time_per_step if avg_time_per_step > 0 else 0
            }

    def print_progress(self, current_step: int, stats: Dict = None):
        """Print progress bar"""
        estimate = self.update(current_step)
        if not estimate:
            return

        # Progress bar
        bar_length = 40
        filled = int(bar_length * estimate['percent'] / 100)
        bar = '█' * filled + '░' * (bar_length - filled)

        # Format times
        elapsed_str = str(timedelta(seconds=int(estimate['elapsed'])))
        remaining_str = str(timedelta(seconds=int(estimate['remaining'])))
        eta_str = estimate['eta'].strftime("%H:%M:%S")

        # Build status line
        status = f"[{bar}] {estimate['percent']:5.1f}% | "
        status += f"{current_step:,}/{self.total_steps:,} steps | "
        status += f"Speed: {estimate['speed']:.2f} steps/s | "
        status += f"Elapsed: {elapsed_str} | "
        status += f"Remaining: {remaining_str} | "
        status += f"ETA: {eta_str}"

        if stats:
            coherence = stats.get('avg_coherence', 0)
            population = stats.get('population_size', 0)
            status += f" | Coh: {coherence:.3f} | Pop: {population}"

        print(f"\r{status}", end='', flush=True)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("GENESIS Phase 4: Monitoring Utilities Demo")
    print("="*70)
    print()

    # Demo monitor
    print("Demo: ExperimentMonitor")
    print("-" * 40)

    monitor = ExperimentMonitor()
    monitor.start()

    for step in range(50):
        # Simulate step
        time.sleep(0.01)

        stats = {
            'avg_coherence': 0.5 + step * 0.001,
            'population_size': 100 + step
        }

        monitor.update(step, stats)

    monitor.stop()
    monitor.print_report()

    # Demo progress estimator
    print("\n\nDemo: ProgressEstimator")
    print("-" * 40)

    estimator = ProgressEstimator(total_steps=100)

    for step in range(100):
        time.sleep(0.01)

        stats = {'avg_coherence': 0.5, 'population_size': 100}
        estimator.print_progress(step, stats)

    print()
    print("\n✓ Monitoring utilities demo complete")
