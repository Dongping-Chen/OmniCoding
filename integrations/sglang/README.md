# SGLang inference

Install SGLang in a GPU environment, then serve a base or merged checkpoint:

```bash
export MODEL_PATH=shuaishuaicdp/Code-X-SFT-27B
export SERVED_MODEL_NAME=shuaishuaicdp/Code-X-SFT-27B
export TP_SIZE=4
bash integrations/sglang/serve.sh
```

The server binds to `127.0.0.1:8080` by default. Exposing it beyond localhost
requires an authentication and network policy chosen by the deployer. Point a
Kira run at it with `OPENAI_API_BASE=http://127.0.0.1:8080/v1`, the LiteLLM
model name `openai/shuaishuaicdp/Code-X-SFT-27B`, and semantic provider
`qwen`. The served name, Kira model suffix, and RL model allowlist must match.
