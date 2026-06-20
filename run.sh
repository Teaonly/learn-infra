set -euo pipefail

MODEL_PATH="/home/teaonly/workspace/qwen3-0.6b"
MODEL_NAME="$(basename "$MODEL_PATH")"
SOCKET_BASE="/tmp/learninfra"

# Export so child processes (e.g. benchmark scripts loading a local tokenizer)
# can resolve the same model path the frontend is serving.
export MODEL_PATH

# Each side cleans only the IPC files it binds, so frontend and backend
# can be started in any order. (Shared mode, num_tokenizer=0.)
case "${1:-}" in
    frontend)
        [[ -d "$MODEL_PATH" ]] || { echo "Error: MODEL_PATH '$MODEL_PATH' not found. Edit $0." >&2; exit 1; }
        rm -f "${SOCKET_BASE}"_1 "${SOCKET_BASE}"_3
        cd "$(pwd)/frontend"
        case "${2:-serve}" in
            serve)
                exec uv run python main.py \
                    --model-path "$MODEL_PATH" \
                    --socket-path "$SOCKET_BASE"
                ;;
            shell)
                exec uv run python main.py \
                    --model-path "$MODEL_PATH" \
                    --socket-path "$SOCKET_BASE" \
                    --shell-mode
                ;;
            *) echo "Usage: $0 frontend [serve|shell]" >&2; exit 64 ;;
        esac
        ;;
    backend)
        [[ -d "$MODEL_PATH" ]] || { echo "Error: MODEL_PATH '$MODEL_PATH' not found. Edit $0." >&2; exit 1; }
        rm -f "${SOCKET_BASE}"_0
        cd "$(pwd)/backend"
        case "${2:-fake}" in
            fake)
                exec uv run python fake.py "$SOCKET_BASE" "$MODEL_PATH"
                ;;
            real)
                echo "Backend is not implemented yet. Use 'fake' mode for testing." >&2
                exit 64
                ;;
            *) echo "Usage: $0 backend [fake|real]" >&2; exit 64 ;;
        esac
        ;;
    bench)
        cd "$(pwd)/benchmark"
        case "${2:-simple}" in
            simple) exec uv run python bench_simple.py "$MODEL_NAME" ;;
            qwen)   exec uv run python bench_qwen.py "$MODEL_NAME" ;;
            *) echo "Usage: $0 bench [simple|qwen]" >&2; exit 64 ;;
        esac
        ;;
    *)
        echo "Usage: $0 {frontend [serve|shell] | backend [fake|real] | bench [simple|qwen]}" >&2
        exit 64
        ;;
esac
