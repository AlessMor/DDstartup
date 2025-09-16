"""
System profiling and optimization utilities for parallel computation.
Automatically profiles system capabilities and optimizes computation settings.
"""

import time
import platform
import psutil
import os


class SystemProfiler:
    """Automatically profile system capabilities and optimize computation settings"""
    
    def __init__(self):
        self.profile = self._profile_system()
        self.optimization = self._optimize_settings()
    
    def _profile_system(self):
        """Profile system hardware and capabilities"""
        profile = {}
        
        # CPU Information
        profile['cpu_count'] = psutil.cpu_count(logical=True)
        profile['cpu_count_physical'] = psutil.cpu_count(logical=False)
        profile['cpu_freq'] = psutil.cpu_freq().current if psutil.cpu_freq() else None
        
        # Memory Information
        memory = psutil.virtual_memory()
        profile['memory_total_gb'] = memory.total / (1024**3)
        profile['memory_available_gb'] = memory.available / (1024**3)
        profile['memory_usage_percent'] = memory.percent
        
        # Platform Information
        profile['platform'] = platform.system()
        profile['machine'] = platform.machine()
        profile['processor'] = platform.processor()
        
        # Performance estimation
        profile['performance_class'] = self._classify_performance(profile)
        
        return profile
    
    def _classify_performance(self, profile):
        """Classify system performance level"""
        score = 0
        
        # CPU scoring
        if profile['cpu_count'] >= 16:
            score += 4
        elif profile['cpu_count'] >= 8:
            score += 3
        elif profile['cpu_count'] >= 4:
            score += 2
        else:
            score += 1
        
        # Memory scoring
        if profile['memory_available_gb'] >= 16:
            score += 4
        elif profile['memory_available_gb'] >= 8:
            score += 3
        elif profile['memory_available_gb'] >= 4:
            score += 2
        else:
            score += 1
        
        # CPU frequency scoring (if available)
        if profile['cpu_freq']:
            if profile['cpu_freq'] >= 3000:
                score += 2
            elif profile['cpu_freq'] >= 2000:
                score += 1
        
        # Classify based on total score
        if score >= 8:
            return "high_performance"
        elif score >= 6:
            return "medium_performance" 
        elif score >= 4:
            return "low_performance"
        else:
            return "minimal_performance"
    
    def _optimize_settings(self):
        """Optimize computation settings based on system profile"""
        settings = {}
        perf_class = self.profile['performance_class']
        
        if perf_class == "high_performance":
            # High-end workstation/server
            settings['n_jobs'] = min(self.profile['cpu_count'] - 2, 16)  # Leave 2 cores for system
            settings['chunk_size'] = 5000
            settings['batch_size'] = 500
            settings['memory_safety_factor'] = 0.8  # Can use more memory
            settings['backend'] = 'loky'  # Can handle process-based parallelism
            settings['use_parallel_chunks'] = True
            
        elif perf_class == "medium_performance":
            # Standard desktop/laptop
            settings['n_jobs'] = min(self.profile['cpu_count'] // 2, 8)
            settings['chunk_size'] = 2000
            settings['batch_size'] = 200
            settings['memory_safety_factor'] = 0.6
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = True
            
        elif perf_class == "low_performance":
            # Older laptop/limited resources
            settings['n_jobs'] = min(self.profile['cpu_count'] // 2, 4)
            settings['chunk_size'] = 1000
            settings['batch_size'] = 100
            settings['memory_safety_factor'] = 0.4
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = False  # Sequential chunks
            
        else:
            # Very limited resources
            settings['n_jobs'] = 1
            settings['chunk_size'] = 500
            settings['batch_size'] = 50
            settings['memory_safety_factor'] = 0.3
            settings['backend'] = 'threading'
            settings['use_parallel_chunks'] = False
        
        # Memory-based adjustments
        available_mem = self.profile['memory_available_gb']
        if available_mem < 2:
            settings['chunk_size'] = min(settings['chunk_size'], 500)
            settings['n_jobs'] = min(settings['n_jobs'], 2)
        elif available_mem < 4:
            settings['chunk_size'] = min(settings['chunk_size'], 1000)
            settings['n_jobs'] = min(settings['n_jobs'], 4)
        
        return settings
    
    def benchmark_single_computation(self, param_combo, input_data, total_time, solve_func=None):
        """Benchmark a single computation to estimate performance"""
        if solve_func is None:
            # Try to find the function in global scope
            import inspect
            frame = inspect.currentframe()
            try:
                while frame:
                    if 'solve_single_combination' in frame.f_globals:
                        solve_func = frame.f_globals['solve_single_combination']
                        break
                    frame = frame.f_back
            finally:
                del frame
            
            if solve_func is None:
                print("Warning: solve_single_combination function not found for benchmark")
                return 0.1, False  # Return default if function not found
        
        start_time = time.time()
        try:
            result = solve_func(param_combo, input_data, total_time)
            end_time = time.time()
            computation_time = end_time - start_time
            if computation_time <= 0:
                print("Warning: Benchmark computation time <= 0, using default estimate")
                return 0.05, True  # Use optimized estimate
            return computation_time, True
        except Exception as e:
            end_time = time.time()
            print(f"Benchmark failed with error: {e}")
            return end_time - start_time, False
    
    def print_system_info(self):
        """Print comprehensive system information"""
        print("=" * 60)
        print("SYSTEM PERFORMANCE PROFILE")
        print("=" * 60)
        print(f"Platform: {self.profile['platform']} ({self.profile['machine']})")
        print(f"CPU Cores: {self.profile['cpu_count']} logical, {self.profile['cpu_count_physical']} physical")
        if self.profile['cpu_freq']:
            print(f"CPU Frequency: {self.profile['cpu_freq']:.0f} MHz")
        print(f"Memory: {self.profile['memory_total_gb']:.1f} GB total, {self.profile['memory_available_gb']:.1f} GB available")
        print(f"Memory Usage: {self.profile['memory_usage_percent']:.1f}%")
        print(f"Performance Class: {self.profile['performance_class'].replace('_', ' ').title()}")
        
        print("\nOPTIMIZED SETTINGS:")
        print(f"Parallel Jobs: {self.optimization['n_jobs']}")
        print(f"Chunk Size: {self.optimization['chunk_size']:,}")
        print(f"Batch Size: {self.optimization['batch_size']:,}")
        print(f"Backend: {self.optimization['backend']}")
        print(f"Parallel Chunks: {self.optimization['use_parallel_chunks']}")
        print(f"Memory Safety: {self.optimization['memory_safety_factor']*100:.0f}%")
        print("=" * 60)


def estimate_computation_time(total_combinations, benchmark_time, n_jobs, parallel_efficiency=0.8):
    """Estimate total computation time"""
    if benchmark_time <= 0:
        return None
    
    sequential_time = total_combinations * benchmark_time
    parallel_time = sequential_time / (n_jobs * parallel_efficiency)
    
    return parallel_time


def format_time_estimate(seconds):
    """Format time estimate in human-readable format"""
    if seconds is None:
        return "Unknown"
    
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"


def get_memory_info():
    """Get available memory in GB"""
    try:
        with open('/proc/meminfo', 'r') as f:
            lines = f.readlines()
        mem_available = None
        for line in lines:
            if 'MemAvailable:' in line:
                mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                break
        if mem_available is None:
            # Fallback if MemAvailable not found
            for line in lines:
                if 'MemFree:' in line:
                    mem_available = int(line.split()[1]) * 1024  # Convert KB to bytes
                    break
        return mem_available / (1024**3)  # Convert to GB
    except:
        return 4.0  # Default fallback


def calculate_optimal_jobs(total_combinations, available_memory_gb):
    """Calculate optimal number of jobs based on memory and combinations"""
    # For very large parameter spaces, use minimal jobs to prevent memory issues
    if total_combinations > 1000000:  # > 1M combinations
        return 2  # Very conservative
    elif total_combinations > 100000:  # > 100K combinations  
        return 3
    elif total_combinations > 10000:   # > 10K combinations
        return 4
    else:
        # Estimate memory per job (conservative estimate)
        memory_per_job_gb = 0.1  # ~100MB per job
        
        # Calculate max jobs based on memory (use only 50% of available memory)
        max_jobs_memory = max(1, int(available_memory_gb * 0.5 / memory_per_job_gb))
        
        # Calculate max jobs based on CPU cores
        max_jobs_cpu = min(4, os.cpu_count())  # Cap at 4 for stability
        
        # Use the minimum of memory and CPU constraints
        optimal_jobs = min(max_jobs_memory, max_jobs_cpu)
        
        return max(1, optimal_jobs)