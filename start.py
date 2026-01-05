"""
FableYard Studio - Main Entry Point

Starts API and Worker processes with shared multiprocessing queues.
Replaces Docker-based orchestration with native Python multiprocessing.
"""

import multiprocessing as mp
import sys
import signal
from pathlib import Path

# Ensure project root in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def start_worker(job_queue: mp.Queue, event_queue: mp.Queue):
    """Worker process entry point"""
    # Import here to avoid issues with multiprocessing on Windows
    # Use absolute import to avoid conflict with Python's built-in 'queue' module
    import sys
    from pathlib import Path
    queue_src = Path(__file__).parent / "queue" / "src"
    sys.path.insert(0, str(queue_src))
    from worker_main import worker_main

    try:
        worker_main(job_queue, event_queue)
    except KeyboardInterrupt:
        print("\n[Worker] Shutting down...")
    except Exception as e:
        print(f"\n[Worker Error] {e}")
        import traceback
        traceback.print_exc()


def start_api(job_queue: mp.Queue, event_queue: mp.Queue):
    """API process entry point (blocks)"""
    import uvicorn
    from api.src.main import create_app

    try:
        app = create_app(job_queue, event_queue)

        print("\n" + "=" * 50)
        print("  FableYard Studio API Starting")
        print("  API: http://localhost:8000")
        print("  Docs: http://localhost:8000/docs")
        print("=" * 50 + "\n")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    except KeyboardInterrupt:
        print("\n[API] Shutting down...")
    except Exception as e:
        print(f"\n[API Error] {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point - starts all services"""
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)

    print("\n" + "=" * 50)
    print("  FableYard Studio")
    print("  Starting API and Worker processes...")
    print("=" * 50 + "\n")

    # Create shared queues
    job_queue = mp.Queue()
    event_queue = mp.Queue()

    # Start worker in separate process
    worker_process = mp.Process(
        target=start_worker,
        args=(job_queue, event_queue),
        name="FableYard-Worker"
    )
    worker_process.start()

    print("[OK] Worker process started (PID: {})".format(worker_process.pid))

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        print("\n\n[SHUTDOWN] Received interrupt signal")
        print("[SHUTDOWN] Stopping worker process...")
        worker_process.terminate()
        worker_process.join(timeout=5)
        if worker_process.is_alive():
            print("[SHUTDOWN] Force killing worker...")
            worker_process.kill()
        print("[SHUTDOWN] Goodbye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start API in main process (blocks until Ctrl+C)
    try:
        start_api(job_queue, event_queue)
    finally:
        print("\n[SHUTDOWN] Cleaning up...")
        if worker_process.is_alive():
            worker_process.terminate()
            worker_process.join(timeout=5)
            if worker_process.is_alive():
                worker_process.kill()
        print("[SHUTDOWN] Complete")


if __name__ == "__main__":
    main()
