set -euo pipefail

MODEL_PATH="/home/teaonly/workspace/qwen3-0.6b"
SOCKET_BASE="/tmp/learninfra"

[[ -d "$MODEL_PATH" ]] || { echo "Error: MODEL_PATH '$MODEL_PATH' not found. Edit $0." >&2; exit 1; }

# Each side cleans only the IPC files it binds, so frontend and backend
# can be started in any order. (Shared mode, num_tokenizer=0.)
case "${1:-}" in
    frontend|shell)
        rm -f "${SOCKET_BASE}"_1 "${SOCKET_BASE}"_3
        cd "$(pwd)/frontend"
        if [[ "${1:-}" == "shell" ]]; then
            exec uv run python main.py \
                --model-path "$MODEL_PATH" \
                --socket-path "$SOCKET_BASE" \
                --shell-mode
        else
            exec uv run python main.py \
                --model-path "$MODEL_PATH" \
                --socket-path "$SOCKET_BASE"
        fi
        ;;
    backend|fake)
        rm -f "${SOCKET_BASE}"_0
        cd "$(pwd)/backend"
        if [[ "${1:-}" == "fake" ]]; then
            exec uv run python fake.py "$SOCKET_BASE" "$MODEL_PATH"
        else
            echo "Backend is not implemented yet. Use 'fake' mode for testing." >&2
        fi
        ;;
    *)
        echo "Usage: $0 {(frontend|shell) (backend|fake)}" >&2
        exit 64
        ;;
esac
