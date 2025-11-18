import asyncio
import functools
import os
import psutil
import threading
import time
from datetime import datetime
from fastapi import Request

LOG_FILE_PATH = "memory_logs.txt"
PROFILING_INTERVAL = 0.01

def log_memory_usage_to_file(func):
    """
    A decorator that logs memory usage before and after a function call,
    and also profiles the peak memory usage during the function's execution
    using a background thread.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Try to get the endpoint path for better logging
        endpoint_info = func.__name__
        for arg in args:
            if isinstance(arg, Request):
                endpoint_info = f"{arg.method} {arg.url.path}"
                break
        
        # --- Profiler thread setup ---
        stop_event = threading.Event()
        memory_samples = []

        def memory_sampler():
            """This function runs in a separate thread to sample memory."""
            while not stop_event.is_set():
                try:
                    memory_samples.append(process.memory_info().rss)
                except Exception:
                    # In case process info can't be read
                    pass
                time.sleep(PROFILING_INTERVAL)

        # --- Execution and Profiling ---
        mem_before = process.memory_info().rss / (1024 ** 2)

        profiler_thread = threading.Thread(target=memory_sampler, daemon=True)
        profiler_thread.start()

        try:
            # Await the actual async function
            result = await func(*args, **kwargs)
        finally:
            # Stop the profiler thread once the function is done
            stop_event.set()
            profiler_thread.join(timeout=1.0)  # Add timeout to prevent hanging

        mem_after = process.memory_info().rss / (1024 ** 2)
        
        # --- Analyze and Log the Results ---
        peak_mem_usage = max(memory_samples) / (1024 ** 2) if memory_samples else mem_after
        mem_diff = mem_after - mem_before

        log_entry = (
            f"[{datetime.utcnow().isoformat()}] [{endpoint_info}] "
            f"Memory Before: {mem_before:.2f} MB | "
            f"After: {mem_after:.2f} MB | "
            f"Diff: {mem_diff:+.2f} MB | "
            f"Peak during call: {peak_mem_usage:.2f} MB\n"
        )

        with open(LOG_FILE_PATH, "a") as f:
            f.write(log_entry)

        return result
    
    return wrapper