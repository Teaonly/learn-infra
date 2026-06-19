# learn-infra frontend

HTTP API server + tokenizer/detokenizer workers for `learn-infra`. Talks to the
(尚未开发的) GPU backend over ZMQ IPC.

## Quick start

```bash
cd frontend
uv sync                                   # 创建 .venv 并安装本包 (editable)
uv run python main.py --help              # 查看 CLI 参数
uv run learn-infra-frontend --help        # 等价的 entry-point 命令
```

启动服务（以本地或 HF 模型为例）：

```bash
uv run python main.py --model-path <path-or-repo-id> --port 1919
```

服务就绪后：

```bash
curl http://127.0.0.1:1919/v1/models
curl http://127.0.0.1:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

> 注意：当前 backend 还没接入，`/generate` / `/v1/chat/completions` 的请求会被
> tokenizer 进程处理并 PUSH 到 `zmq_backend_addr`，但没有任何 backend 接收 ——
> 因此 response 会卡住。HTTP 路由 + tokenizer 子进程 + ZMQ 通路可以先验证。

## 前后端通过 ZMQ IPC 通信

默认情况下 ZMQ IPC 文件用 PID 后缀（`/tmp/learninfra.pid={pid}_0` 等），多个 frontend
实例不会冲突，但外部进程没法预先知道路径。

要让 backend 单独启动并连上 frontend，启动 frontend 时显式指定 `--socket-path`：

```bash
# 终端 1：起 frontend
uv run python main.py --model-path <model> --socket-path /tmp/learninfra-run1
# 启动日志会打印 4 个实际使用的 ipc:// 地址，例如：
#   tokenizer → backend : ipc:///tmp/learninfra-run1_0
#   backend  → detok    : ipc:///tmp/learninfra-run1_1
#   detok    → HTTP API : ipc:///tmp/learninfra-run1_3
#   HTTP API → tokenizer: ipc:///tmp/learninfra-run1_4   (仅 num_tokenizer>0)

# 终端 2：起 backend (尚未实现，伪示例)
# backend 启动时用同样的 socket-path，bind/correctly connect 上述地址即可
```

socket 槽位约定：

| 地址                       | 方向                  | 谁来 bind           |
|----------------------------|-----------------------|---------------------|
| `{socket_path}_0`          | tokenizer → backend   | backend (PULL)      |
| `{socket_path}_1`          | backend → detokenizer | tokenizer (PULL)    |
| `{socket_path}_3`          | detokenizer → HTTP    | HTTP API (PULL)     |
| `{socket_path}_4`          | HTTP → tokenizer      | HTTP API (PUSH-bind 当 num_tokenizer>0) |

## Architecture

```
HTTP API  ──PUSH──▶  Tokenizer worker  ──PUSH──▶  (backend, 未实现)
                     (detokenizer 共享)            │
                                                    ▼
HTTP API  ◀──PULL──  Tokenizer worker  ◀──PUSH──  (backend, 未实现)
```

- `frontend.server` — FastAPI app、CLI args、`launch_server` 入口
- `frontend.tokenizer` — tokenize / detokenize multiprocessing worker
- `frontend.message` — msgpack + 自定义 dataclass 序列化协议
- `frontend.utils` — ZMQ 队列封装、logger、HF tokenizer loader
- `frontend.backend` — backend 桥接（占位）

ZMQ 地址在 `ServerArgs` (`frontend/server/args.py`) 里生成：未指定 `--socket-path`
时用 PID 后缀（避免冲突）；指定后用固定路径（便于前后端独立启动）。
