"""learn-infra frontend CLI entry point.

Launches the HTTP API server together with tokenizer/detokenizer worker
processes. The GPU backend is started separately (see the project-root
``run.sh``) and rendezvous with this process over ZMQ IPC.

Usage:
    python main.py --model-path PATH [options]

Required:
    --model-path PATH, --model PATH
        Local folder containing the model weights (must already exist on
        disk — no remote download is performed).

Common options (full list in frontend/server/args.py):
    --host HOST            HTTP bind host.        (default: 127.0.0.1)
    --port PORT            HTTP bind port.        (default: 1919)
    --socket-path PATH     ZMQ IPC base path shared with the backend process.
                           Defaults to a PID-suffixed /tmp/learninfra* path.
    --num-tokenizer N      Number of tokenizer worker processes (>=1, default 1).
                           The detokenizer always runs as a separate process.
    --shell-mode           Interactive shell mode (implies silent output).

Examples:
    python main.py --model-path /data/models/my-model
    python main.py --model-path /data/models/llama-3-8b --port 8080
    python main.py --model-path ./ckpt --socket-path /tmp/learninfra
"""

from frontend.server.launch import launch_server

if __name__ == "__main__":
    launch_server()
