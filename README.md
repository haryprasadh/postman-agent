# Postman Agent (local)

Build **Postman Collection v2.1** JSON from API docs / OpenAPI / GitHub links using an AI model. Runs on **your machine only** (default: `127.0.0.1`).

**API keys** (OpenAI, Anthropic, Groq, OpenRouter, etc.) are **not** loaded from `.env`. You enter them in the **web UI**; they are **encrypted in the browser** and sent to your local server on each request.

---

## Are we using OpenRouter?

**No by default.**  
With a normal **OpenAI** key from the app, requests go to **OpenAI** (`api.openai.com`) and **`gpt-4o-mini`** unless you change **`AI_BASE_URL`** / **`AI_MODEL`** in `.env`.

OpenRouter (and Groq, etc.) are **optional**: the server **infers** the provider from common key shapes, or you set **`AI_BASE_URL` + `AI_MODEL`** in `.env` (no secrets there).

---

## New user setup (minimal)

1. **Install Node.js 18+**
2. In the project folder:

   ```bash
   npm install
   ```

   Optional: `cp .env.example .env` if you want to tweak **`AI_BASE_URL`**, models, **`PORT`**, etc. Keys stay out of `.env`.

3. Start the app:

   ```bash
   npm run dev
   ```

   Open **http://localhost:3000**

4. When the **API keys** modal opens, paste your **AI** key (required) and optionally your **Postman** key, then **Save**.

5. Restart the server after changing **`.env`** (provider defaults only — not keys).

---

## How it works (short)

1. **Keys** live in **browser localStorage** (encrypted), not in server environment variables.
2. The server **auto-detects the provider** from the key (and optional **`AI_VENDOR`** in `.env`):
   - **Anthropic** (`sk-ant-…` or `AI_VENDOR=anthropic`): native **Messages API** with Claude **web_fetch** / **web_search** server tools where supported.
   - **Everyone else** (OpenAI, Groq, OpenRouter, etc.): **OpenAI-compatible** `POST …/v1/chat/completions` via the **OpenAI SDK** + configurable `baseURL`.
3. For OpenAI-compatible builds, the model can call a **`fetch_url`** tool that runs **on your computer** to load docs/OpenAPI pages (localhost-only by default).
4. **Postman push** uses an optional Postman key from the same modal.

---

## Auto-detected keys (from the key you paste in the app)

If **`AI_BASE_URL`** is unset in `.env`, the server picks a default **from the key** (OpenAI-compatible path only):

| Key looks like | Base URL | Default model |
|----------------|----------|----------------|
| Starts with **`sk-ant-`** (Anthropic) | `https://api.anthropic.com` (native API) | `claude-sonnet-4-20250514` (build/chat); override with `ANTHROPIC_MODEL_*` |
| Starts with `gsk_` (Groq) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Starts with `sk-or-v1-` or `sk-or-` (OpenRouter) | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| Other `sk-…` (e.g. OpenAI) | `https://api.openai.com/v1` | `gpt-4o-mini` |

Override anytime with **`AI_BASE_URL`** and **`AI_MODEL`**, or **`AI_VENDOR`** (`groq`, `openrouter`, `together`, `anthropic`) if detection is wrong. For Anthropic-only overrides, use **`ANTHROPIC_MODEL`**, **`ANTHROPIC_MODEL_BUILD`**, **`ANTHROPIC_MODEL_CHAT`**, or **`ANTHROPIC_MODEL_VALIDATE`**.

OpenRouter requests send default `HTTP-Referer` / `X-Title` headers (customize with `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_NAME` if needed).

---

## Environment reference (no secrets)

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `AI_BASE_URL` | No | Override API root (OpenAI-compatible path only) |
| `AI_MODEL` | No | Override model id (OpenAI-compatible path; also fallback for Anthropic if set) |
| `AI_VENDOR` | No | `groq` / `openrouter` / `together` / `anthropic` if auto-detect fails |
| `ANTHROPIC_MODEL` / `ANTHROPIC_MODEL_BUILD` / `ANTHROPIC_MODEL_CHAT` / `ANTHROPIC_MODEL_VALIDATE` | No | Anthropic model ids |
| `AI_MAX_TOKENS_*`, `AI_MAX_TOOL_ROUNDS` | No | Token / tool-round limits |
| `PORT` | No | Default `3000` |
| `HOST` | No | Default `127.0.0.1` (local only) |
| `NODE_ENV`, `TRUST_PROXY` | No | Production / reverse-proxy behavior |

---

## Security

- Default **`HOST=127.0.0.1`**: not reachable from other devices.
- **Keys** are in the **browser**; the local server sees them only on each request. Do not expose this server to untrusted networks if you care about key confidentiality.
- **`fetch_url`** (OpenAI-compatible builds) runs on your machine — only use on networks you trust.

---

## API (debugging)

- `GET /health`
- `GET /api/config` — resolved default `aiProvider`, `aiBaseUrl`, `aiModel` from `.env` overrides (no secrets; `aiConfigured` is always false)
