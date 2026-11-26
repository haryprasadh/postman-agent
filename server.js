import 'dotenv/config';
import express from 'express';
import Anthropic from '@anthropic-ai/sdk';
import nodeFetch, { Headers, Request, Response, FormData } from 'node-fetch';
import OpenAI from 'openai';

// OpenAI SDK uses `Headers` / `Request` / etc. globally; Node < 18 (or some runtimes) omit them.
if (typeof globalThis.fetch !== 'function') {
  globalThis.fetch = nodeFetch;
}
if (typeof globalThis.Headers === 'undefined') {
  globalThis.Headers = Headers;
}
if (typeof globalThis.Request === 'undefined') {
  globalThis.Request = Request;
}
if (typeof globalThis.Response === 'undefined') {
  globalThis.Response = Response;
}
if (typeof globalThis.FormData === 'undefined') {
  globalThis.FormData = FormData;
}

const fetch = globalThis.fetch.bind(globalThis);
import { v4 as uuidv4 } from 'uuid';
import path from 'path';
import { fileURLToPath } from 'url';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const IS_PROD = process.env.NODE_ENV === 'production';

/** Generic header for any OpenAI-compatible provider */
const AI_API_KEY_HEADER = 'x-ai-api-key';
/** Legacy header (still accepted) */
const LEGACY_AI_HEADER = 'x-anthropic-api-key';
const POSTMAN_HEADER = 'x-postman-api-key';

const MAX_SOURCE_LEN = 2048;
const MAX_NOTES_LEN = 4000;
const MAX_MESSAGE_LEN = 16000;
const MAX_HISTORY_MESSAGES = 30;
const MAX_HISTORY_CONTENT = 120_000;
const MAX_BUILD_HISTORY_ITEMS = 6;
const MAX_BUILD_HISTORY_TEXT = 1200;

const AI_MAX_TOKENS_BUILD = Math.min(
  Number.parseInt(process.env.AI_MAX_TOKENS_BUILD || '4096', 10) || 4096,
  8192
);
const AI_MAX_TOKENS_CHAT = Math.min(
  Number.parseInt(process.env.AI_MAX_TOKENS_CHAT || '8192', 10) || 8192,
  16384
);
const MAX_TOOL_ROUNDS = Math.min(
  Number.parseInt(process.env.AI_MAX_TOOL_ROUNDS || '8', 10) || 8,
  12
);

const app = express();

const HOST = process.env.HOST?.trim() || '127.0.0.1';

if (process.env.TRUST_PROXY === '1') {
  app.set('trust proxy', 1);
}

app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'unsafe-inline'"],
        // Default script-src-attr is 'none'; UI uses addEventListener (no inline handlers)
        styleSrc: ["'self'", "'unsafe-inline'"],
        imgSrc: ["'self'", 'data:', 'https:'],
        connectSrc: ["'self'"],
        frameAncestors: ["'none'"],
        baseUri: ["'self'"],
        formAction: ["'self'"],
      },
    },
    crossOriginEmbedderPolicy: false,
  })
);

app.use(express.json({ limit: '4mb' }));

const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: IS_PROD ? 400 : 2000,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests' },
});

const buildLimiter = rateLimit({
  windowMs: 60 * 60 * 1000,
  max: IS_PROD ? 40 : 300,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many collection builds, try again later' },
});

const validateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: IS_PROD ? 60 : 300,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many validation attempts' },
});

app.use('/api', apiLimiter);

app.get('/health', (_req, res) => {
  res.set('Cache-Control', 'no-store');
  res.json({ ok: true, uptime: process.uptime() });
});

app.get('/api/config', (_req, res) => {
  res.set('Cache-Control', 'no-store');
  const provider = inferAiProvider('');
  let aiBaseUrl;
  let aiModel;
  if (provider === 'anthropic') {
    aiBaseUrl = 'https://api.anthropic.com';
    aiModel = resolveAnthropicModel('build');
  } else {
    const cfg = resolveAiConfig('');
    aiBaseUrl = cfg.baseURL;
    aiModel = cfg.model;
  }
  res.json({
    /** Always false — API keys are supplied from the app (or X-AI-API-Key), not .env */
    aiConfigured: false,
    /** @deprecated use aiConfigured */
    anthropicConfigured: false,
    postmanConfigured: false,
    aiProvider: provider,
    aiBaseUrl,
    aiModel,
  });
});

function getAiApiKey(req) {
  return req.get(AI_API_KEY_HEADER)?.trim() || req.get(LEGACY_AI_HEADER)?.trim() || null;
}

/** @returns {'anthropic' | 'openai_compatible'} */
function inferAiProvider(apiKey) {
  const key = (apiKey || '').trim();
  if (!key) return 'openai_compatible';
  if (process.env.AI_VENDOR?.trim().toLowerCase() === 'anthropic') return 'anthropic';
  if (key.startsWith('sk-ant-')) return 'anthropic';
  return 'openai_compatible';
}

function resolveAnthropicModel(kind) {
  const fromEnv =
    kind === 'validate'
      ? process.env.ANTHROPIC_MODEL_VALIDATE?.trim()
      : kind === 'chat'
        ? process.env.ANTHROPIC_MODEL_CHAT?.trim() || process.env.ANTHROPIC_MODEL?.trim()
        : process.env.ANTHROPIC_MODEL_BUILD?.trim() ||
          process.env.ANTHROPIC_MODEL?.trim() ||
          process.env.AI_MODEL?.trim();
  if (fromEnv) return fromEnv;
  if (kind === 'validate') return 'claude-3-5-haiku-20241022';
  return 'claude-sonnet-4-20250514';
}

function makeAnthropicClient(apiKey) {
  return new Anthropic({ apiKey });
}

function getPostmanKey(req) {
  return req.get(POSTMAN_HEADER)?.trim() || null;
}

/**
 * Resolves API base URL + model. If AI_BASE_URL is unset, picks a sensible default from the key
 * (e.g. Groq gsk_, OpenRouter sk-or-v1-, else OpenAI) when the client sends a key.
 */
function resolveAiConfig(apiKey) {
  const envBase = process.env.AI_BASE_URL?.trim();
  const envModel = process.env.AI_MODEL?.trim();
  const vendor = process.env.AI_VENDOR?.trim().toLowerCase();

  const key = apiKey || '';
  let base = envBase;
  let model = envModel;

  if (!base) {
    if (vendor === 'groq' || key.startsWith('gsk_')) {
      base = 'https://api.groq.com/openai/v1';
      if (!model) model = 'llama-3.3-70b-versatile';
    } else if (
      vendor === 'openrouter' ||
      /^sk-or-v1-/i.test(key) ||
      (key.startsWith('sk-or-') && !key.startsWith('sk-proj-'))
    ) {
      base = 'https://openrouter.ai/api/v1';
      if (!model) model = 'openai/gpt-4o-mini';
    } else if (vendor === 'together') {
      base = 'https://api.together.xyz/v1';
      if (!model) model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo';
    } else {
      base = 'https://api.openai.com/v1';
      if (!model) model = 'gpt-4o-mini';
    }
  }

  const baseURL = base.replace(/\/$/, '');
  if (!model) model = 'gpt-4o-mini';

  const useOpenRouterHeaders = baseURL.includes('openrouter.ai');
  const defaultHeaders = useOpenRouterHeaders
    ? {
        'HTTP-Referer': process.env.OPENROUTER_HTTP_REFERER || 'http://localhost:3000',
        'X-Title': process.env.OPENROUTER_APP_NAME || 'Postman Agent',
      }
    : undefined;

  return { baseURL, model, defaultHeaders };
}

function makeOpenAIClient(apiKey) {
  const { baseURL, model, defaultHeaders } = resolveAiConfig(apiKey);
  const client = new OpenAI({
    apiKey,
    baseURL,
    defaultHeaders,
    fetch,
  });
  return { client, model, baseURL };
}

function clientError(message, status = 400) {
  const err = new Error(message);
  err.status = status;
  return err;
}

function validateSourceInput(source) {
  const s = source?.trim();
  if (!s) throw clientError('Source is required');
  if (s.length > MAX_SOURCE_LEN) throw clientError('Source URL is too long');
  // Source can be either a URL or free text ("Stripe billing API with webhooks").
  try {
    const u = new URL(s);
    if (!['http:', 'https:'].includes(u.protocol)) {
      throw clientError('Source URL must use http or https');
    }
  } catch {
    // Non-URL input is allowed.
  }
  return s;
}

function validateNotes(notes) {
  if (!notes?.trim()) return '';
  const t = notes.trim();
  if (t.length > MAX_NOTES_LEN) throw clientError('Notes are too long');
  return t;
}

function sanitizeChatHistory(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const item of raw.slice(-MAX_HISTORY_MESSAGES)) {
    if (!item || typeof item !== 'object') continue;
    const { role, content } = item;
    if (role !== 'user' && role !== 'assistant') continue;
    if (typeof content !== 'string' || !content.trim()) continue;
    const c = content.length > MAX_HISTORY_CONTENT ? content.slice(0, MAX_HISTORY_CONTENT) : content;
    out.push({ role, content: c });
  }
  return out.slice(-MAX_HISTORY_MESSAGES);
}

function validateMessage(message) {
  if (!message?.trim()) throw clientError('Message required');
  const m = message.trim();
  if (m.length > MAX_MESSAGE_LEN) throw clientError('Message is too long');
  return m;
}

function sanitizeBuildHistory(raw) {
  if (!Array.isArray(raw)) return [];
  const out = [];
  for (const item of raw.slice(-MAX_BUILD_HISTORY_ITEMS)) {
    if (!item || typeof item !== 'object') continue;
    const source = typeof item.source === 'string' ? item.source.trim().slice(0, 500) : '';
    const notes = typeof item.notes === 'string' ? item.notes.trim().slice(0, 500) : '';
    const summary =
      typeof item.summary === 'string' ? item.summary.trim().slice(0, MAX_BUILD_HISTORY_TEXT) : '';
    if (!source && !summary) continue;
    out.push({
      source,
      notes,
      summary,
      endpointCount: Number(item.endpointCount) || 0,
      folderCount: Number(item.folderCount) || 0,
      name: typeof item.name === 'string' ? item.name.trim().slice(0, 120) : '',
    });
  }
  return out;
}

function logError(err) {
  if (IS_PROD) {
    console.error('[error]', err?.name || 'Error', err?.message || err);
  } else {
    console.error(err);
  }
}

function shortId() {
  return uuidv4().split('-')[0];
}

function logger(scope, reqId = '-') {
  const prefix = `[${scope}][${reqId}]`;
  return {
    info: (...args) => console.log(prefix, ...args),
    warn: (...args) => console.warn(prefix, ...args),
    error: (...args) => console.error(prefix, ...args),
  };
}

// ─── Prompts ──────────────────────────────────────────────────────────────────

const BUILD_SYSTEM_PROMPT = `You are an expert Postman collection builder. Analyze any API source and generate Postman collections ONLY from verifiable evidence in fetched content.

## Process
1. Use the **fetch_url** tool to retrieve the user's source URL and any follow-up URLs (OpenAPI/Swagger JSON or YAML, raw GitHub files, HTML docs, etc.).
2. Use the **web_search** tool when docs are blocked/missing to discover public API evidence (official docs, OpenAPI files, GitHub repos, and technical posts).
3. If HTML, look for Swagger UI links, OpenAPI spec links, API docs links — fetch them with fetch_url.
4. If GitHub, look for README, openapi.yaml/json, swagger.yaml/json, API route files — use fetch_url on raw or API URLs.
5. Analyze every endpoint, method, parameter, header, body schema, and auth requirement.
6. Generate a complete Postman Collection v2.1 JSON ONLY when evidence is sufficient.

## Evidence rules (strict)
- DO NOT guess endpoints, paths, parameters, auth, or request bodies.
- DO NOT invent APIs from domain assumptions or common patterns.
- Every request in the collection must be traceable to fetched evidence.
- Public sources (forums/blogs) can guide discovery, but include endpoints only when endpoint+method are explicitly shown in fetched evidence.
- If evidence is insufficient or blocked (403/404/CAPTCHA), do NOT output collection JSON.
- In insufficient-evidence cases, return a concise explanation of what was inaccessible and which direct URLs are needed.

## Rules
- Use \`{{base_url}}\` as a collection-level variable for the API base URL
- Use \`{{api_key}}\` for API key auth, \`{{token}}\` for Bearer tokens
- Organize endpoints into logical folders (Authentication, Users, Products, Orders, etc.)
- Every request gets a descriptive name and description
- Include realistic example bodies for POST/PUT/PATCH
- Add all required headers (Content-Type, Authorization, Accept)
- Include query parameters with descriptions
- For path params: use :param in raw URL and add to url.variable array

## Output
Respond with analysis, then end with EXACTLY ONE JSON code block:

\`\`\`json
{
  "info": { "_postman_id": "REPLACE_WITH_UUID", "name": "...", "description": "...", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json" },
  "item": [ { "name": "Folder", "item": [ { "name": "Request", "request": { "method": "GET", "header": [], "url": { "raw": "{{base_url}}/path", "host": ["{{base_url}}"], "path": ["path"] } } } ] } ],
  "variable": [ { "key": "base_url", "value": "https://api.example.com", "type": "string" } ]
}
\`\`\``;

const BUILD_SYSTEM_PROMPT_ANTHROPIC = `You are an expert Postman collection builder. Analyze any API source and generate Postman collections ONLY from verifiable evidence in fetched content.

## Process
1. Use **web_fetch** to retrieve the user's source URL and any follow-up URLs (OpenAPI/Swagger JSON or YAML, raw GitHub files, HTML docs, etc.). Use **web_search** if you need to locate a public spec or doc.
2. If HTML, look for Swagger UI links, OpenAPI spec links, API docs links — fetch them.
3. If GitHub, look for README, openapi.yaml/json, swagger.yaml/json, API route files.
4. Analyze every endpoint, method, parameter, header, body schema, and auth requirement.
5. Generate a complete Postman Collection v2.1 JSON ONLY when evidence is sufficient.

## Evidence rules (strict)
- DO NOT guess endpoints, paths, parameters, auth, or request bodies.
- DO NOT invent APIs from domain assumptions or common patterns.
- Every request in the collection must be traceable to fetched evidence.
- If evidence is insufficient or blocked (403/404/CAPTCHA), do NOT output collection JSON.
- In insufficient-evidence cases, return a concise explanation of what was inaccessible and which direct URLs are needed.

## Rules
- Use \`{{base_url}}\` as a collection-level variable for the API base URL
- Use \`{{api_key}}\` for API key auth, \`{{token}}\` for Bearer tokens
- Organize endpoints into logical folders (Authentication, Users, Products, Orders, etc.)
- Every request gets a descriptive name and description
- Include realistic example bodies for POST/PUT/PATCH
- Add all required headers (Content-Type, Authorization, Accept)
- Include query parameters with descriptions
- For path params: use :param in raw URL and add to url.variable array

## Output
Respond with analysis, then end with EXACTLY ONE JSON code block:

\`\`\`json
{
  "info": { "_postman_id": "REPLACE_WITH_UUID", "name": "...", "description": "...", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json" },
  "item": [ { "name": "Folder", "item": [ { "name": "Request", "request": { "method": "GET", "header": [], "url": { "raw": "{{base_url}}/path", "host": ["{{base_url}}"], "path": ["path"] } } } ] } ],
  "variable": [ { "key": "base_url", "value": "https://api.example.com", "type": "string" } ]
}
\`\`\``;

const FETCH_URL_TOOL = {
  type: 'function',
  function: {
    name: 'fetch_url',
    description:
      'Download an http(s) URL and return status, content-type, and body text (truncated if very large). Use for API docs, OpenAPI JSON, Swagger, GitHub raw files, HTML pages.',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'Full http or https URL to fetch' },
      },
      required: ['url'],
    },
  },
};

const WEB_SEARCH_TOOL = {
  type: 'function',
  function: {
    name: 'web_search',
    description:
      'Search the public web for API docs/specs/examples. Returns top result URLs and snippets for follow-up fetching.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
      },
      required: ['query'],
    },
  },
};

const CHAT_SYSTEM_PROMPT = (collection) => `You are a Postman collection assistant. A collection has already been built and you help the user understand, refine, and extend it.

You can:
- Explain any endpoint, folder, or request in the collection
- Add new endpoints or folders when asked
- Modify existing requests
- Explain how to use authentication
- Clarify request bodies, query params, and response formats

Current collection (JSON):
\`\`\`json
${JSON.stringify(collection, null, 2).slice(0, 40000)}
\`\`\`

If the user asks you to add or modify endpoints, include a complete updated collection JSON block at the end of your response:
\`\`\`json
{ ...updated collection... }
\`\`\`

Be concise and helpful. Keep requests under the existing run folder(s) unless the user explicitly asks to reorganize. When showing URLs or endpoints inline, use \`code formatting\`.`;

// ─── fetch_url (server-side; local app — do not expose on untrusted networks) ─

const FETCH_MAX_BYTES = Math.min(
  Number.parseInt(process.env.FETCH_MAX_BYTES || '120000', 10) || 120000,
  250000
);
const FETCH_TIMEOUT_MS = 28_000;
const WEB_SEARCH_TIMEOUT_MS = 20_000;
const WEB_SEARCH_MAX_RESULTS = Math.min(
  Number.parseInt(process.env.WEB_SEARCH_MAX_RESULTS || '8', 10) || 8,
  12
);
const TOOL_RESULT_CHAR_LIMIT = Math.min(
  Number.parseInt(process.env.TOOL_RESULT_CHAR_LIMIT || '40000', 10) || 40000,
  80000
);
const TOOL_CONTEXT_BUDGET_CHARS = Math.min(
  Number.parseInt(process.env.TOOL_CONTEXT_BUDGET_CHARS || '160000', 10) || 160000,
  320000
);

async function safeFetchUrl(rawUrl, log = null) {
  if (!rawUrl || typeof rawUrl !== 'string') {
    return { text: 'Error: invalid url parameter', ok: false, status: 0, contentType: '', bytes: 0 };
  }
  let u;
  try {
    u = new URL(rawUrl.trim());
  } catch {
    return { text: 'Error: malformed URL', ok: false, status: 0, contentType: '', bytes: 0 };
  }
  if (!['http:', 'https:'].includes(u.protocol)) {
    return {
      text: 'Error: only http and https are allowed',
      ok: false,
      status: 0,
      contentType: '',
      bytes: 0,
    };
  }

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    log?.info('fetch_url:start', u.toString());
    const r = await fetch(u.toString(), {
      signal: ctrl.signal,
      redirect: 'follow',
      headers: {
        'User-Agent': 'Postman-Agent/1.0 (local)',
        Accept: 'text/html,application/json,text/plain,*/*;q=0.8',
      },
    });
    const ct = r.headers.get('content-type') || '';
    const buf = await r.arrayBuffer();
    const truncated = buf.byteLength > FETCH_MAX_BYTES;
    const slice = truncated ? buf.slice(0, FETCH_MAX_BYTES) : buf;
    let text = new TextDecoder('utf-8', { fatal: false }).decode(slice);
    if (truncated) text += '\n\n[Response truncated by Postman Agent for size]';
    log?.info('fetch_url:done', `status=${r.status} bytes=${buf.byteLength} truncated=${truncated}`, ct);
    return {
      text: `HTTP ${r.status}\nContent-Type: ${ct}\n\n${text}`,
      ok: r.ok,
      status: r.status,
      contentType: ct,
      bytes: buf.byteLength,
      truncated,
    };
  } catch (e) {
    const reason = e.name === 'AbortError' ? 'timeout' : e.message;
    log?.warn('fetch_url:error', reason);
    return {
      text: `Error fetching URL: ${reason}`,
      ok: false,
      status: 0,
      contentType: '',
      bytes: 0,
    };
  } finally {
    clearTimeout(timer);
  }
}

function stripHtml(input = '') {
  return String(input)
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<\/?[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, ' ')
    .trim();
}

async function safeWebSearch(query, log = null) {
  const q = String(query || '').trim();
  if (!q) return 'Error: empty search query';
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), WEB_SEARCH_TIMEOUT_MS);
  try {
    const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(q)}`;
    log?.info('web_search:start', q);
    const r = await fetch(url, {
      signal: ctrl.signal,
      headers: {
        'User-Agent': 'Postman-Agent/1.0 (local)',
        Accept: 'text/html,application/xhtml+xml',
      },
    });
    const html = await r.text();
    const matches = [
      ...html.matchAll(/<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi),
    ];
    const results = [];
    for (const m of matches.slice(0, WEB_SEARCH_MAX_RESULTS)) {
      const href = decodeDuckduckgoRedirect(m[1]);
      const title = stripHtml(m[2]).slice(0, 180);
      if (!href || !title) continue;
      results.push(`- ${title}\n  URL: ${href}`);
    }
    if (!results.length) {
      for (const m of html.matchAll(/<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]{1,220}?)<\/a>/gi)) {
        const href = decodeDuckduckgoRedirect(m[1]);
        const title = stripHtml(m[2]).slice(0, 180);
        if (!href || !title || /duckduckgo\.com/i.test(href)) continue;
        results.push(`- ${title}\n  URL: ${href}`);
        if (results.length >= WEB_SEARCH_MAX_RESULTS) break;
      }
    }
    log?.info('web_search:done', `status=${r.status} results=${results.length}`);
    if (!results.length) return `Web search query: ${q}\nNo parseable results.`;
    return `Web search query: ${q}\nTop results:\n${results.join('\n')}`;
  } catch (e) {
    const reason = e.name === 'AbortError' ? 'timeout' : e.message;
    log?.warn('web_search:error', reason);
    return `Error searching web: ${reason}`;
  } finally {
    clearTimeout(timer);
  }
}

function emitTextChunks(fullText, send, chunkSize = 96) {
  for (let i = 0; i < fullText.length; i += chunkSize) {
    send({ type: 'chunk', content: fullText.slice(i, i + chunkSize) });
  }
}

function zeroUsage() {
  return { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
}

function addUsage(base, input = 0, output = 0, total = 0) {
  base.inputTokens += Number(input) || 0;
  base.outputTokens += Number(output) || 0;
  base.totalTokens += Number(total) || 0;
}

function inferPricePer1K(model = '') {
  const m = String(model).toLowerCase();
  if (m.includes('gpt-4o-mini')) return { inPer1K: 0.00015, outPer1K: 0.0006 };
  if (m.includes('gpt-4o')) return { inPer1K: 0.005, outPer1K: 0.015 };
  if (m.includes('claude-sonnet-4')) return { inPer1K: 0.003, outPer1K: 0.015 };
  if (m.includes('claude-3-5-haiku')) return { inPer1K: 0.0008, outPer1K: 0.004 };
  return null;
}

function estimateCostUsd(model, usage) {
  const pricing = inferPricePer1K(model);
  if (!pricing) return null;
  const inCost = (usage.inputTokens / 1000) * pricing.inPer1K;
  const outCost = (usage.outputTokens / 1000) * pricing.outPer1K;
  return Number((inCost + outCost).toFixed(6));
}

function withBuildRunFolder(collection) {
  const stamp = new Date().toISOString().replace('T', ' ').slice(0, 19);
  const runFolderName = `Build ${stamp}`;
  const items = Array.isArray(collection.item) ? collection.item : [];
  collection.item = [{ name: runFolderName, item: items }];
  return runFolderName;
}

function extractCurlCommands(text) {
  if (!text || typeof text !== 'string') return [];
  const oneLine = [...text.matchAll(/curl\s+https?:\/\/[^\n]+/gi)].map((m) => m[0]);
  const quoted = [...text.matchAll(/curl\s+['"]https?:\/\/[^'"]+['"][^\n]*/gi)].map((m) => m[0]);
  const multiline = [...text.matchAll(/curl[\s\S]*?(?=\n\s*\n|$)/gi)].map((m) => m[0]);
  const blocks = [...oneLine, ...quoted, ...multiline]
    .map((s) => s.replace(/\\\n/g, ' ').trim())
    .filter((s) => /^curl\b/i.test(s));
  return [...new Set(blocks)].slice(0, 100);
}

function tokenizeShellLike(command) {
  const tokens = [];
  const re = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^\s]+)/g;
  let m;
  while ((m = re.exec(command))) {
    tokens.push((m[1] || m[2] || m[3] || '').replace(/\\'/g, "'").replace(/\\"/g, '"'));
  }
  return tokens;
}

function parseCurlCommand(command) {
  const tokens = tokenizeShellLike(command);
  if (!tokens.length || tokens[0] !== 'curl') return null;
  let method = 'GET';
  let url = null;
  const headers = [];
  let body = null;
  for (let i = 1; i < tokens.length; i++) {
    const t = tokens[i];
    if ((t === '-X' || t === '--request') && tokens[i + 1]) {
      method = String(tokens[++i]).toUpperCase();
      continue;
    }
    if ((t === '-H' || t === '--header') && tokens[i + 1]) {
      const raw = tokens[++i];
      const idx = raw.indexOf(':');
      if (idx > 0) {
        headers.push({ key: raw.slice(0, idx).trim(), value: raw.slice(idx + 1).trim(), type: 'text' });
      }
      continue;
    }
    if ((t === '-d' || t === '--data' || t === '--data-raw' || t === '--data-binary') && tokens[i + 1]) {
      body = tokens[++i];
      if (method === 'GET') method = 'POST';
      continue;
    }
    if (/^https?:\/\//i.test(t)) {
      url = t;
    }
  }
  if (!url) return null;
  let u;
  try {
    u = new URL(url);
  } catch {
    return null;
  }
  const pathParts = u.pathname.split('/').filter(Boolean);
  const query = [...u.searchParams.entries()].map(([key, value]) => ({ key, value }));
  return {
    name: `${method} ${u.pathname || '/'}`,
    request: {
      method,
      header: headers,
      url: {
        raw: u.toString(),
        protocol: u.protocol.replace(':', ''),
        host: u.hostname.split('.'),
        path: pathParts,
        ...(query.length ? { query } : {}),
      },
      ...(body
        ? {
            body: {
              mode: 'raw',
              raw: body,
            },
          }
        : {}),
    },
  };
}

function collectionFromCurlCommands(commands) {
  const items = commands.map(parseCurlCommand).filter(Boolean);
  if (!items.length) return null;
  return {
    info: {
      name: 'Collection from captured cURL',
      description:
        'Generated directly from browser/network captured cURL commands. Endpoints are evidence-based from captured traffic.',
      schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json',
    },
    item: [{ name: 'Captured Requests', item: items }],
    variable: [],
  };
}

function extractHarPayload(text) {
  if (!text || typeof text !== 'string') return null;
  const trimmed = text.trim();
  const tryParse = (raw) => {
    try {
      const obj = JSON.parse(raw);
      if (obj?.log?.entries && Array.isArray(obj.log.entries)) return obj;
      return null;
    } catch {
      return null;
    }
  };
  const direct = tryParse(trimmed);
  if (direct) return direct;

  // Allow fenced JSON blocks in notes/source.
  const fenced = trimmed.match(/```json\s*([\s\S]*?)\s*```/i) || trimmed.match(/```\s*([\s\S]*?)\s*```/i);
  if (fenced) {
    const parsed = tryParse(fenced[1]);
    if (parsed) return parsed;
  }
  return null;
}

function normalizeHarHeaders(headers = []) {
  return (headers || [])
    .filter((h) => h && typeof h.name === 'string')
    .map((h) => ({
      key: h.name,
      value: typeof h.value === 'string' ? h.value : String(h.value || ''),
      type: 'text',
    }));
}

function parseHarEntries(harObj) {
  const entries = Array.isArray(harObj?.log?.entries) ? harObj.log.entries : [];
  const out = [];
  for (const e of entries) {
    const req = e?.request;
    if (!req || typeof req.url !== 'string' || typeof req.method !== 'string') continue;
    let u;
    try {
      u = new URL(req.url);
    } catch {
      continue;
    }
    const method = req.method.toUpperCase();
    const headers = normalizeHarHeaders(req.headers);
    let body;
    const mime = req.postData?.mimeType || '';
    const text = req.postData?.text;
    if (typeof text === 'string' && text.length) {
      body = {
        mode: 'raw',
        raw: text,
        ...(mime ? { options: { raw: { language: /json/i.test(mime) ? 'json' : 'text' } } } : {}),
      };
    }
    out.push({
      method,
      url: u.toString(),
      headers,
      body,
      evidence: 'har_capture',
    });
  }
  return out;
}

function collectionFromHar(harObj) {
  const requests = parseHarEntries(harObj);
  if (!requests.length) return null;
  const items = requests.map((r) => {
    const u = new URL(r.url);
    return {
      name: u.pathname || '/',
      request: {
        method: r.method,
        header: r.headers || [],
        url: {
          raw: r.url,
          protocol: u.protocol.replace(':', ''),
          host: u.hostname.split('.'),
          path: u.pathname.split('/').filter(Boolean),
          query: [...u.searchParams.entries()].map(([key, value]) => ({ key, value })),
        },
        ...(r.body ? { body: r.body } : {}),
      },
    };
  });
  return {
    info: {
      name: 'Collection from HAR capture',
      description:
        'Generated from browser HAR (DevTools network export). Endpoints are evidence-based from captured traffic.',
      schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json',
    },
    item: [{ name: 'HAR Captured Requests', item: items }],
    variable: [],
  };
}

function toAbsoluteUrl(raw, baseUrl) {
  const s = String(raw || '').trim();
  if (!s) return null;
  try {
    return new URL(s, baseUrl).toString();
  } catch {
    return null;
  }
}

function extractScriptUrlsFromHtml(html, baseUrl) {
  const scripts = [];
  for (const m of String(html || '').matchAll(/<script[^>]+src=["']([^"']+)["'][^>]*>/gi)) {
    const abs = toAbsoluteUrl(m[1], baseUrl);
    if (abs) scripts.push(abs);
  }
  return [...new Set(scripts)].slice(0, 24);
}

function extractLikelyJsUrlsFromJs(jsText, baseUrl) {
  const out = [];
  const text = String(jsText || '');
  for (const m of text.matchAll(/["'`]([^"'`]+\.js(?:\?[^"'`]*)?)["'`]/g)) {
    const raw = m[1];
    if (!raw || raw.length > 300) continue;
    const abs = toAbsoluteUrl(raw, baseUrl);
    if (!abs) continue;
    if (!/^https?:\/\//i.test(abs)) continue;
    out.push(abs);
  }
  return [...new Set(out)].slice(0, 40);
}

function extractApiLikeUrlsFromText(text, baseUrl) {
  const out = [];
  const s = String(text || '');
  for (const m of s.matchAll(/https?:\/\/[^\s"'`<>)]+/g)) {
    if (/api|graphql|v\d\//i.test(m[0])) out.push(m[0]);
  }
  for (const m of s.matchAll(/["'`]((?:\/|\.\/|\.\.\/)[^"'`]*(?:api|graphql)[^"'`]*)["'`]/gi)) {
    const abs = toAbsoluteUrl(m[1], baseUrl);
    if (abs) out.push(abs);
  }
  return [...new Set(out)].slice(0, 80);
}

function decodeDuckduckgoRedirect(href) {
  try {
    const u = new URL(href);
    const uddg = u.searchParams.get('uddg');
    if (uddg) return decodeURIComponent(uddg);
  } catch {
    /* ignore */
  }
  return href;
}

function extractMethodUrlPairsFromText(text, baseUrl) {
  const out = [];
  const s = String(text || '');
  for (const m of s.matchAll(/\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+(https?:\/\/[^\s"'`<>)]+|\/[^\s"'`<>)]+)/gi)) {
    const method = m[1].toUpperCase();
    const url = toAbsoluteUrl(m[2], baseUrl);
    if (url) out.push({ method, url, evidence: 'text_method_url' });
  }
  for (const m of s.matchAll(
    /method\s*[:=]\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`][\s\S]{0,160}?url\s*[:=]\s*["'`]([^"'`]+)["'`]/gi
  )) {
    const method = m[1].toUpperCase();
    const url = toAbsoluteUrl(m[2], baseUrl);
    if (url) out.push({ method, url, evidence: 'text_kv_method_url' });
  }
  for (const m of s.matchAll(
    /url\s*[:=]\s*["'`]([^"'`]+)["'`][\s\S]{0,160}?method\s*[:=]\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`]/gi
  )) {
    const method = m[2].toUpperCase();
    const url = toAbsoluteUrl(m[1], baseUrl);
    if (url) out.push({ method, url, evidence: 'text_kv_url_method' });
  }
  const dedup = new Map();
  for (const r of out) {
    const key = `${r.method} ${r.url}`;
    if (!dedup.has(key)) dedup.set(key, r);
  }
  return [...dedup.values()].slice(0, 180);
}

function extractObservedRequestsFromJs(jsText, baseUrl) {
  const text = String(jsText || '');
  const out = [];

  // fetch('/api/x', { method: 'POST' })
  for (const m of text.matchAll(
    /fetch\(\s*["'`]([^"'`]+)["'`]\s*,\s*\{[\s\S]{0,260}?method\s*:\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`]/gi
  )) {
    const url = toAbsoluteUrl(m[1], baseUrl);
    if (url) out.push({ method: m[2].toUpperCase(), url, evidence: 'fetch(method)' });
  }

  // axios.post('/api/x', ...)
  for (const m of text.matchAll(/axios\.(get|post|put|patch|delete|options|head)\(\s*["'`]([^"'`]+)["'`]/gi)) {
    const url = toAbsoluteUrl(m[2], baseUrl);
    if (url) out.push({ method: m[1].toUpperCase(), url, evidence: 'axios.verb' });
  }

  // $.ajax({ url: '/api/x', type: 'PUT' })
  for (const m of text.matchAll(
    /\$\.(?:ajax|get|post)\(\s*\{[\s\S]{0,260}?url\s*:\s*["'`]([^"'`]+)["'`][\s\S]{0,260}?(?:type|method)\s*:\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`]/gi
  )) {
    const url = toAbsoluteUrl(m[1], baseUrl);
    if (url) out.push({ method: m[2].toUpperCase(), url, evidence: 'jquery.ajax' });
  }

  // axios({ method: 'post', url: '/api/x' })
  for (const m of text.matchAll(
    /axios\(\s*\{[\s\S]{0,260}?method\s*:\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`][\s\S]{0,260}?url\s*:\s*["'`]([^"'`]+)["'`]/gi
  )) {
    const url = toAbsoluteUrl(m[2], baseUrl);
    if (url) out.push({ method: m[1].toUpperCase(), url, evidence: 'axios(config)' });
  }
  for (const m of text.matchAll(
    /axios\(\s*\{[\s\S]{0,260}?url\s*:\s*["'`]([^"'`]+)["'`][\s\S]{0,260}?method\s*:\s*["'`](GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)["'`]/gi
  )) {
    const url = toAbsoluteUrl(m[1], baseUrl);
    if (url) out.push({ method: m[2].toUpperCase(), url, evidence: 'axios(config)' });
  }

  const dedup = new Map();
  for (const r of out) {
    const key = `${r.method} ${r.url}`;
    if (!dedup.has(key)) dedup.set(key, r);
  }
  return [...dedup.values()].slice(0, 120);
}

function collectionFromObservedRequests(observed) {
  const valid = (observed || []).filter((r) => r?.method && r?.url);
  if (!valid.length) return null;
  const items = valid.map((r) => {
    const u = new URL(r.url);
    const pathName = u.pathname || '/';
    return {
      // UI already renders request method separately; keep name path-focused.
      name: pathName,
      request: {
        method: r.method,
        header: [],
        url: {
          raw: r.url,
          protocol: u.protocol.replace(':', ''),
          host: u.hostname.split('.'),
          path: u.pathname.split('/').filter(Boolean),
          query: [...u.searchParams.entries()].map(([key, value]) => ({ key, value })),
        },
        description: `Observed from frontend bundle (${r.evidence}).`,
      },
    };
  });
  return {
    info: {
      name: 'Collection from observed frontend API calls',
      description:
        'Built from explicit fetch/axios/ajax calls discovered in the target app scripts. No guessed endpoints included.',
      schema: 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json',
    },
    item: [{ name: 'Observed API Calls', item: items }],
    variable: [],
  };
}

function rootDomain(hostname) {
  const parts = String(hostname || '').toLowerCase().split('.').filter(Boolean);
  if (parts.length <= 2) return parts.join('.');
  return parts.slice(-2).join('.');
}

function looksTemplatedUrl(url) {
  const s = String(url || '');
  return /(\$\{[^}]+\}|%7B[^%]+%7D)/i.test(s);
}

function isLowSignalTrackingPath(pathname) {
  const p = String(pathname || '').toLowerCase();
  return (
    /utm|tracking|analytics|pixel|survey\/web|qevents|gtm/.test(p) ||
    p.includes('/web/utm_details')
  );
}

function isLikelyBackendEvidence(method, url, sourceUrl) {
  if (!method || !url) return false;
  let parsed;
  let src;
  try {
    parsed = new URL(url);
    src = new URL(sourceUrl);
  } catch {
    return false;
  }
  if (looksTemplatedUrl(url)) return false;
  if (isLowSignalTrackingPath(parsed.pathname)) return false;
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return false;

  const host = parsed.hostname.toLowerCase();
  const srcHost = src.hostname.toLowerCase();
  const sameRoot = rootDomain(host) === rootDomain(srcHost);
  const firstPartyHost = sameRoot || host.startsWith('api.');

  const path = parsed.pathname.toLowerCase();
  const backendishPath =
    /\/api(\/|$)|\/v[0-9]+(\/|$)|\/auth(\/|$)|\/login(\/|$)|\/token(\/|$)|\/graphql(\/|$)/.test(path);
  const backendishMethod = ['POST', 'PUT', 'PATCH', 'DELETE', 'GET'].includes(String(method).toUpperCase());
  return firstPartyHost && backendishMethod && (backendishPath || sameRoot);
}

function filterObservedRequestsForBackend(observed, sourceUrl) {
  const cleaned = (observed || []).filter((r) => r?.method && r?.url);
  const high = cleaned.filter((r) => isLikelyBackendEvidence(r.method, r.url, sourceUrl));
  if (high.length) return high;
  // If strict filter yields nothing, retain only non-templated first-party URLs.
  return cleaned.filter((r) => {
    try {
      const u = new URL(r.url);
      const s = new URL(sourceUrl);
      return !looksTemplatedUrl(r.url) && rootDomain(u.hostname) === rootDomain(s.hostname);
    } catch {
      return false;
    }
  });
}

const PREFETCH_SPEC_PATHS = [
  '/openapi.json',
  '/swagger.json',
  '/v3/api-docs',
  '/api/openapi.json',
  '/api/swagger.json',
  '/.well-known/openapi.json',
  '/api-docs',
];
const PREFETCH_AUX_PATHS = ['/robots.txt', '/sitemap.xml', '/manifest.json', '/sw.js', '/service-worker.js'];

async function prefetchEvidenceHints(source, send, log) {
  if (!/^https?:\/\//i.test(source)) return { hints: '', observedRequests: [] };
  const lines = [];
  const observedRequests = [];
  const root = await safeFetchUrl(source, log);
  lines.push(`source ${source} -> status=${root.status || 'n/a'} ct=${root.contentType || 'unknown'}`);
  let base;
  try {
    base = new URL(source);
  } catch {
    return { hints: lines.join('\n'), observedRequests };
  }
  if (root.ok && /html/i.test(root.contentType || '')) {
    const scriptQueue = extractScriptUrlsFromHtml(root.text, base.toString());
    const visitedScripts = new Set();
    lines.push(`html scripts discovered=${scriptQueue.length}`);
    while (scriptQueue.length && visitedScripts.size < 48) {
      const scriptUrl = scriptQueue.shift();
      if (!scriptUrl || visitedScripts.has(scriptUrl)) continue;
      visitedScripts.add(scriptUrl);
      send({ type: 'status', message: `Scanning script: ${scriptUrl.slice(0, 72)}...` });
      const jsRes = await safeFetchUrl(scriptUrl, log);
      if (jsRes.ok && /(javascript|text\/plain|application\/json|html)/i.test(jsRes.contentType || '')) {
        const found = extractObservedRequestsFromJs(jsRes.text, base.toString());
        if (found.length) {
          lines.push(`${scriptUrl} -> observed_api_calls=${found.length}`);
          observedRequests.push(...found);
        }
        const textPairs = extractMethodUrlPairsFromText(jsRes.text, base.toString());
        if (textPairs.length) {
          lines.push(`${scriptUrl} -> explicit_method_url_pairs=${textPairs.length}`);
          observedRequests.push(...textPairs);
        }
        const discoveredJs = extractLikelyJsUrlsFromJs(jsRes.text, base.toString());
        for (const u of discoveredJs) {
          if (!visitedScripts.has(u) && !scriptQueue.includes(u) && scriptQueue.length < 40) {
            scriptQueue.push(u);
          }
        }
      }
    }
    lines.push(`scripts_scanned=${visitedScripts.size}`);
  }
  for (const path of PREFETCH_AUX_PATHS) {
    const auxUrl = new URL(path, base).toString();
    send({ type: 'status', message: `Probing aux: ${auxUrl.slice(0, 72)}...` });
    const aux = await safeFetchUrl(auxUrl, log);
    lines.push(`${auxUrl} -> status=${aux.status || 'n/a'} ct=${aux.contentType || 'unknown'} bytes=${aux.bytes || 0}`);
    if (aux.ok && aux.text) {
      observedRequests.push(...extractMethodUrlPairsFromText(aux.text, base.toString()));
    }
  }
  const candidates = PREFETCH_SPEC_PATHS.map((p) => new URL(p, base).toString());
  for (const url of candidates.slice(0, 6)) {
    send({ type: 'status', message: `Probing: ${url.slice(0, 72)}...` });
    const r = await safeFetchUrl(url, log);
    lines.push(`${url} -> status=${r.status || 'n/a'} ct=${r.contentType || 'unknown'} bytes=${r.bytes || 0}`);
  }
  const dedup = new Map();
  for (const r of observedRequests) {
    const key = `${r.method} ${r.url}`;
    if (!dedup.has(key)) dedup.set(key, r);
  }
  let normalizedObserved = [...dedup.values()].slice(0, 120);
  if (normalizedObserved.length) {
    lines.push(`observed_request_signatures=${normalizedObserved.length}`);
  } else {
    // Extra resilience: discover public references and mine explicit API-like URLs.
    send({ type: 'status', message: 'Searching public web for API references...' });
    const host = base.host;
    const searchText = await safeWebSearch(`${host} API documentation endpoints`, log);
    lines.push(`public_search_performed_for=${host}`);
    const candidateUrls = [];
    for (const m of String(searchText).matchAll(/URL:\s*(https?:\/\/\S+)/g)) {
      const u = m[1].trim();
      if (!candidateUrls.includes(u)) candidateUrls.push(u);
      if (candidateUrls.length >= 6) break;
    }
    for (const u of candidateUrls) {
      send({ type: 'status', message: `Reviewing public ref: ${u.slice(0, 72)}...` });
      const ref = await safeFetchUrl(u, log);
      if (!ref.ok) continue;
      observedRequests.push(...extractMethodUrlPairsFromText(ref.text, base.toString()));
      const apiLike = extractApiLikeUrlsFromText(ref.text, base.toString());
      if (apiLike.length) {
        lines.push(`${u} -> api_like_urls=${apiLike.length}`);
        for (const apiUrl of apiLike.slice(0, 20)) {
          // Method unknown from plain text refs -> do not convert to request here.
          // Keep as evidence hints for model/tool follow-up only.
          lines.push(`candidate_api_url ${apiUrl}`);
        }
      }
      const curls = extractCurlCommands(ref.text);
      if (curls.length) {
        const fromCurl = curls.map(parseCurlCommand).filter(Boolean);
        if (fromCurl.length) {
          lines.push(`${u} -> curl_requests=${fromCurl.length}`);
          for (const req of fromCurl.slice(0, 30)) {
            const raw = req?.request?.url?.raw;
            const method = req?.request?.method;
            if (raw && method) {
              observedRequests.push({ method, url: raw, evidence: 'public_ref_curl' });
            }
          }
        }
      }
    }
    const dedupAfterSearch = new Map();
    for (const r of observedRequests) {
      const key = `${r.method} ${r.url}`;
      if (!dedupAfterSearch.has(key)) dedupAfterSearch.set(key, r);
    }
    normalizedObserved = [...dedupAfterSearch.values()].slice(0, 120);
  }
  return { hints: lines.join('\n').slice(0, 5000), observedRequests: normalizedObserved };
}

async function runBuildWithFetchTool(client, model, send, source, notes, buildHistory, evidenceHints = '') {
  const historyText = buildHistory?.length
    ? `\nPrevious successful builds (reuse patterns where useful):\n${buildHistory
        .map(
          (h, i) =>
            `${i + 1}. source=${h.source || '-'}; name=${h.name || '-'}; endpoints=${h.endpointCount}; folders=${h.folderCount}; summary=${h.summary || '-'}`
        )
        .join('\n')}`
    : '';
  const sourceLooksUrl = /^https?:\/\//i.test(source);
  const userContent = `${sourceLooksUrl ? `Source URL: ${source}` : `Source topic: ${source}`}\n${
    notes ? `Instructions: ${notes}` : 'Build all possible API endpoints in a logical folder structure.'
  }\n${
    sourceLooksUrl
      ? 'This source may be a live app link; inspect linked docs/specs and discover request patterns from public network calls and documentation where possible.'
      : 'Use only verifiable evidence from accessible documentation/content the user provided.'
  }${historyText}${evidenceHints ? `\n\nPrefetched evidence hints:\n${evidenceHints}` : ''}`;

  const messages = [
    { role: 'system', content: BUILD_SYSTEM_PROMPT },
    { role: 'user', content: userContent },
  ];

  let fullText = '';
  const usage = zeroUsage();
  let accumulatedToolChars = 0;

  for (let round = 0; round < MAX_TOOL_ROUNDS; round++) {
    send({ type: 'status', message: `Analyzing (step ${round + 1})...` });

    const completion = await client.chat.completions.create({
      model,
      messages,
      tools: [FETCH_URL_TOOL, WEB_SEARCH_TOOL],
      tool_choice: 'auto',
      parallel_tool_calls: false,
      max_tokens: AI_MAX_TOKENS_BUILD,
    });

    const choice = completion.choices[0];
    if (!choice?.message) {
      throw new Error('Empty model response');
    }
    addUsage(
      usage,
      completion.usage?.prompt_tokens,
      completion.usage?.completion_tokens,
      completion.usage?.total_tokens
    );

    const msg = choice.message;
    messages.push(msg);

    const toolCalls = msg.tool_calls;
    if (toolCalls?.length) {
      for (const tc of toolCalls) {
        if (tc.type !== 'function') continue;
        const name = tc.function?.name;
        let args = {};
        try {
          args = JSON.parse(tc.function.arguments || '{}');
        } catch {
          args = {};
        }
        if (name === 'fetch_url') {
          const url = args.url;
          send({
            type: 'status',
            message: `Fetching: ${String(url || '(invalid)').slice(0, 72)}...`,
          });
          const result = await safeFetchUrl(url);
          let toolContent = result.text || '';
          if (toolContent.length > TOOL_RESULT_CHAR_LIMIT) {
            toolContent =
              toolContent.slice(0, TOOL_RESULT_CHAR_LIMIT) +
              '\n\n[Tool result truncated by Postman Agent context guardrail]';
          }
          accumulatedToolChars += toolContent.length;
          if (accumulatedToolChars > TOOL_CONTEXT_BUDGET_CHARS) {
            send({
              type: 'warning',
              message:
                'Tool context budget reached; stopping additional fetch payload growth to avoid model context overflow.',
            });
            throw clientError(
              'Context too large from fetched docs. Provide a direct OpenAPI/Swagger URL (or narrower source) and retry.',
              400
            );
          }
          send({
            type: 'debug',
            message: `fetch_url result → status=${result.status || 'n/a'} bytes=${result.bytes || 0} ct=${result.contentType || 'unknown'} chars=${toolContent.length} budget=${accumulatedToolChars}/${TOOL_CONTEXT_BUDGET_CHARS}`,
          });
          messages.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: toolContent,
          });
        } else if (name === 'web_search') {
          const query = String(args.query || '').slice(0, 300);
          send({
            type: 'status',
            message: `Searching web: ${query.slice(0, 64)}...`,
          });
          const result = await safeWebSearch(query);
          let clipped = result;
          if (clipped.length > TOOL_RESULT_CHAR_LIMIT) {
            clipped =
              clipped.slice(0, TOOL_RESULT_CHAR_LIMIT) +
              '\n\n[Web search result truncated by Postman Agent context guardrail]';
          }
          accumulatedToolChars += clipped.length;
          if (accumulatedToolChars > TOOL_CONTEXT_BUDGET_CHARS) {
            send({
              type: 'warning',
              message:
                'Tool context budget reached; stopping additional search payload growth to avoid model context overflow.',
            });
            throw clientError(
              'Context too large from discovery/search data. Provide a direct OpenAPI/Swagger URL (or narrower source) and retry.',
              400
            );
          }
          send({
            type: 'debug',
            message: `web_search result → chars=${clipped.length} budget=${accumulatedToolChars}/${TOOL_CONTEXT_BUDGET_CHARS}`,
          });
          messages.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: clipped,
          });
        } else {
          messages.push({
            role: 'tool',
            tool_call_id: tc.id,
            content: `Error: unknown tool ${name}`,
          });
        }
      }
      continue;
    }

    fullText = typeof msg.content === 'string' ? msg.content : '';
    break;
  }

  if (!fullText.trim()) {
    send({ type: 'status', message: 'Asking model for final answer...' });
    const final = await client.chat.completions.create({
      model,
      messages,
      tool_choice: 'none',
      max_tokens: AI_MAX_TOKENS_BUILD,
    });
    const m = final.choices[0]?.message;
    addUsage(
      usage,
      final.usage?.prompt_tokens,
      final.usage?.completion_tokens,
      final.usage?.total_tokens
    );
    fullText = typeof m?.content === 'string' ? m.content : '';
  }

  emitTextChunks(fullText, send);
  return { fullText, usage };
}

const ANTHROPIC_BUILD_MAX = Math.min(AI_MAX_TOKENS_BUILD, 64000);

async function runBuildAnthropic(client, model, send, source, notes, buildHistory, evidenceHints = '') {
  const historyText = buildHistory?.length
    ? `\nPrevious successful builds (reuse patterns where useful):\n${buildHistory
        .map(
          (h, i) =>
            `${i + 1}. source=${h.source || '-'}; name=${h.name || '-'}; endpoints=${h.endpointCount}; folders=${h.folderCount}; summary=${h.summary || '-'}`
        )
        .join('\n')}`
    : '';
  const userMessage = `Source: ${source.trim()}${
    notes ? `\nInstructions: ${notes}` : '\nBuild all possible API endpoints in a logical folder structure.'
  }\nThis source may be a live app link; inspect linked docs/specs and discover request patterns from public network calls and documentation where possible.${historyText}${
    evidenceHints ? `\n\nPrefetched evidence hints:\n${evidenceHints}` : ''
  }`;
  let messages = [{ role: 'user', content: userMessage }];
  let fullText = '';
  const usage = zeroUsage();
  let pass = 0;

  while (pass < 6) {
    pass++;
    send({ type: 'status', message: `Analyzing (step ${pass})...` });

    const stream = client.messages.stream({
      model,
      max_tokens: ANTHROPIC_BUILD_MAX,
      thinking: { type: 'adaptive' },
      system: BUILD_SYSTEM_PROMPT_ANTHROPIC,
      tools: [
        { type: 'web_fetch_20260209', name: 'web_fetch' },
        { type: 'web_search_20260209', name: 'web_search' },
      ],
      messages,
    });

    for await (const event of stream) {
      if (event.type === 'content_block_start') {
        if (event.content_block.type === 'server_tool_use') {
          send({ type: 'status', message: `Using ${event.content_block.name}...` });
        }
      } else if (event.type === 'content_block_delta') {
        if (event.delta.type === 'text_delta') {
          fullText += event.delta.text;
          send({ type: 'chunk', content: event.delta.text });
        } else if (event.delta.type === 'input_json_delta') {
          const m = event.delta.partial_json?.match(/"url"\s*:\s*"([^"]{10,70})/);
          if (m) send({ type: 'status', message: `Fetching: ${m[1]}...` });
        }
      }
    }

    const response = await stream.finalMessage();
    addUsage(
      usage,
      response?.usage?.input_tokens,
      response?.usage?.output_tokens,
      (response?.usage?.input_tokens || 0) + (response?.usage?.output_tokens || 0)
    );
    if (response.stop_reason === 'pause_turn') {
      messages.push({ role: 'assistant', content: response.content });
      continue;
    }
    break;
  }

  return { fullText, usage };
}

async function runChatAnthropic(client, model, send, collection, history, message) {
  const messages = [
    ...history.map((h) => ({ role: h.role, content: h.content })),
    { role: 'user', content: message },
  ];

  const stream = client.messages.stream({
    model,
    max_tokens: AI_MAX_TOKENS_CHAT,
    system: CHAT_SYSTEM_PROMPT(collection || {}),
    messages,
  });

  let fullText = '';
  for await (const event of stream) {
    if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
      fullText += event.delta.text;
      send({ type: 'chunk', content: event.delta.text });
    }
  }
  return fullText;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function countEndpoints(items = []) {
  let n = 0;
  for (const item of items) {
    if (item.request) n++;
    if (item.item) n += countEndpoints(item.item);
  }
  return n;
}

function extractCollection(text) {
  function pickCollection(obj) {
    if (!obj || typeof obj !== 'object' || Array.isArray(obj)) return null;
    if (obj.info && obj.item) return obj;
    if (obj.collection && typeof obj.collection === 'object') {
      const nested = obj.collection;
      if (nested.info && nested.item) return nested;
    }
    return null;
  }

  const fencedBlocks = [...text.matchAll(/```(?:json|javascript|js)?\s*([\s\S]*?)\s*```/gim)];
  for (const block of fencedBlocks) {
    try {
      const parsed = JSON.parse(block[1]);
      const picked = pickCollection(parsed);
      if (picked) return picked;
    } catch {
      /* ignore */
    }
  }

  const starts = [];
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') starts.push(i);
  }
  for (const start of starts) {
    let depth = 0;
    let inString = false;
    let escaped = false;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (ch === '\\') {
          escaped = true;
        } else if (ch === '"') {
          inString = false;
        }
        continue;
      }
      if (ch === '"') {
        inString = true;
        continue;
      }
      if (ch === '{') depth++;
      if (ch === '}') depth--;
      if (depth === 0) {
        const candidate = text.slice(start, i + 1);
        if (!candidate.includes('"info"') || !candidate.includes('"item"')) break;
        try {
          const parsed = JSON.parse(candidate);
          const picked = pickCollection(parsed);
          if (picked) return picked;
        } catch {
          /* ignore */
        }
        break;
      }
    }
  }

  return null;
}

function normalizeRequestUrl(url) {
  if (!url) return null;
  if (typeof url === 'string') {
    return { raw: url };
  }
  if (typeof url === 'object') {
    return url;
  }
  return null;
}

function normalizeItemNode(node) {
  if (!node || typeof node !== 'object') return null;
  const name = typeof node.name === 'string' && node.name.trim() ? node.name.trim() : null;

  if (Array.isArray(node.item)) {
    const children = node.item.map(normalizeItemNode).filter(Boolean);
    if (!children.length) return null;
    return {
      name: name || 'Folder',
      item: children,
      ...(typeof node.description === 'string' ? { description: node.description } : {}),
    };
  }

  if (node.request && typeof node.request === 'object') {
    const method = typeof node.request.method === 'string' ? node.request.method.toUpperCase() : 'GET';
    const url = normalizeRequestUrl(node.request.url);
    if (!url) return null;
    return {
      name: name || `${method} request`,
      request: {
        method,
        url,
        ...(Array.isArray(node.request.header) ? { header: node.request.header } : {}),
        ...(node.request.body ? { body: node.request.body } : {}),
        ...(node.request.auth ? { auth: node.request.auth } : {}),
        ...(typeof node.request.description === 'string'
          ? { description: node.request.description }
          : {}),
      },
      ...(Array.isArray(node.response) ? { response: node.response } : {}),
    };
  }

  return null;
}

function normalizeCollectionShape(rawCollection) {
  if (!rawCollection || typeof rawCollection !== 'object' || Array.isArray(rawCollection)) {
    return null;
  }

  const rawItems = Array.isArray(rawCollection.item) ? rawCollection.item : [];
  const item = rawItems.map(normalizeItemNode).filter(Boolean);
  const info = rawCollection.info && typeof rawCollection.info === 'object' ? rawCollection.info : {};
  const variable = Array.isArray(rawCollection.variable) ? rawCollection.variable : [];

  return {
    ...rawCollection,
    info: {
      ...info,
      name: typeof info.name === 'string' && info.name.trim() ? info.name.trim() : 'API Collection',
    },
    item,
    variable,
  };
}

function summarizePostmanErrorBody(errorBody) {
  if (!errorBody || typeof errorBody !== 'object') return '';
  const details = Array.isArray(errorBody.error?.details)
    ? errorBody.error.details
    : Array.isArray(errorBody.details)
      ? errorBody.details
      : [];
  if (!details.length) return '';
  const msgs = details
    .map((d) => {
      if (!d || typeof d !== 'object') return null;
      const m = typeof d.message === 'string' ? d.message.trim() : '';
      const p = typeof d.path === 'string' ? ` (${d.path})` : '';
      return m ? `${m}${p}` : null;
    })
    .filter(Boolean)
    .slice(0, 3);
  return msgs.length ? msgs.join(' | ') : '';
}

function sseResponder(res) {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
  return (data) => {
    if (!res.writableEnded) res.write(`data: ${JSON.stringify(data)}\n\n`);
  };
}

// ─── /api/build ───────────────────────────────────────────────────────────────

app.post('/api/build', buildLimiter, async (req, res) => {
  const reqId = shortId();
  const log = logger('build', reqId);
  const buildStartedAt = Date.now();
  let source;
  let notes;
  let buildHistory;
  try {
    source = validateSourceInput(req.body?.source);
    notes = validateNotes(req.body?.notes);
    buildHistory = sanitizeBuildHistory(req.body?.buildHistory);
  } catch (e) {
    return res.status(e.status || 400).json({ error: e.message });
  }

  const postmanApiKey = getPostmanKey(req);
  const send = sseResponder(res);
  const debug = (message) => {
    send({ type: 'debug', message: `[${reqId}] ${message}` });
    log.info(message);
  };
  let usage = zeroUsage();
  let modelUsed = '';

  try {
    const harPayload = extractHarPayload(`${source}\n${notes}`);
    if (harPayload) {
      send({ type: 'status', message: 'Parsing HAR capture...' });
      const harRequests = parseHarEntries(harPayload);
      debug(`captured_har_requests=${harRequests.length}`);
      const filteredHar = filterObservedRequestsForBackend(harRequests, source);
      debug(`captured_har_requests_filtered=${filteredHar.length}/${harRequests.length}`);
      let collection = normalizeCollectionShape(collectionFromObservedRequests(filteredHar));
      if (!collection) {
        // fallback to full HAR if backend filter is too strict
        collection = normalizeCollectionShape(collectionFromHar(harPayload));
      }
      if (!collection) {
        throw clientError('Could not parse valid requests from HAR payload', 400);
      }
      if (!collection.info) collection.info = {};
      collection.info._postman_id = uuidv4();
      collection.info.schema = 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json';
      const runFolderName = withBuildRunFolder(collection);
      const name = collection.info?.name || 'API Collection';
      const endpointCount = countEndpoints(collection.item);
      send({
        type: 'warning',
        message: 'Built directly from HAR capture (browser network export).',
      });
      const elapsedMs = Date.now() - buildStartedAt;
      send({
        type: 'metrics',
        elapsedMs,
        usage,
        model: modelUsed,
        estimatedCostUsd: estimateCostUsd(modelUsed, usage),
        runFolderName,
      });
      send({ type: 'done', collection, name, endpointCount, postman: null });
      return res.end();
    }

    const capturedCurlCommands = extractCurlCommands(`${source}\n${notes}`);
    debug(`captured_curl_commands=${capturedCurlCommands.length}`);
    if (capturedCurlCommands.length) {
      send({ type: 'status', message: `Parsing ${capturedCurlCommands.length} captured cURL command(s)...` });
      const parsed = collectionFromCurlCommands(capturedCurlCommands);
      let collection = normalizeCollectionShape(parsed);
      if (!collection) {
        throw clientError('Could not parse cURL commands from source/notes', 400);
      }
      if (!collection.info) collection.info = {};
      collection.info._postman_id = uuidv4();
      collection.info.schema = 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json';
      const runFolderName = withBuildRunFolder(collection);
      const name = collection.info?.name || 'API Collection';
      const endpointCount = countEndpoints(collection.item);

      let postmanResult = null;
      if (postmanApiKey && endpointCount > 0) {
        send({ type: 'status', message: 'Pushing to Postman...' });
        try {
          const h = { 'Content-Type': 'application/json', 'X-Api-Key': postmanApiKey };
          const wsRes = await fetch('https://api.getpostman.com/workspaces', { headers: h });
          let workspaceId = null;
          if (wsRes.ok) {
            const ws = await wsRes.json();
            const personal = (ws.workspaces || []).find((w) => w.type === 'personal') || ws.workspaces?.[0];
            workspaceId = personal?.id;
          }
          const url = workspaceId
            ? `https://api.getpostman.com/collections?workspace=${workspaceId}`
            : 'https://api.getpostman.com/collections';
          const pr = await fetch(url, { method: 'POST', headers: h, body: JSON.stringify({ collection }) });
          const pb = await pr.json();
          if (!pr.ok) {
            const detailMsg = summarizePostmanErrorBody(pb);
            send({
              type: 'warning',
              message: `Postman (${pr.status}): ${pb.error?.message || pb.message || 'unknown error'}${
                detailMsg ? ` — ${detailMsg}` : ''
              }`,
            });
          } else {
            const uid = pb.collection?.uid || pb.collection?.id;
            postmanResult = { uid, url: uid ? `https://go.postman.co/collections/${uid}` : null };
            send({ type: 'postman_success', uid, url: postmanResult.url });
          }
        } catch (err) {
          send({ type: 'warning', message: `Postman push failed: ${err.message}` });
        }
      }

      const elapsedMs = Date.now() - buildStartedAt;
      send({
        type: 'metrics',
        elapsedMs,
        usage,
        model: modelUsed,
        estimatedCostUsd: estimateCostUsd(modelUsed, usage),
        runFolderName,
      });
      send({ type: 'done', collection, name, endpointCount, postman: postmanResult });
      return res.end();
    }

    const apiKey = getAiApiKey(req);
    if (!apiKey) {
      return res.status(401).json({
        error:
          'AI API key required. Add your key in the app or send the X-AI-API-Key header. (Or paste cURL commands from browser network and retry.)',
      });
    }

    const provider = inferAiProvider(apiKey);
    debug(`source=${source}`);
    debug(`provider=${provider}`);
    let fullText;
    let evidenceHints = '';
    let observedRequests = [];
    if (/^https?:\/\//i.test(source)) {
      send({ type: 'status', message: 'Running resilient pre-probes...' });
      const prefetch = await prefetchEvidenceHints(source, send, log);
      evidenceHints = prefetch?.hints || '';
      observedRequests = Array.isArray(prefetch?.observedRequests) ? prefetch.observedRequests : [];
      if (evidenceHints) {
        debug(`prefetch_hints=${evidenceHints.slice(0, 180).replace(/\s+/g, ' ')}`);
      }
      if (observedRequests.length) {
        debug(`observed_requests=${observedRequests.length}`);
      }
    }
    if (provider === 'anthropic') {
      const anthropic = makeAnthropicClient(apiKey);
      const model = resolveAnthropicModel('build');
      modelUsed = model;
      send({ type: 'status', message: 'Connecting to Anthropic…' });
      debug(`model=${model}`);
      const result = await runBuildAnthropic(
        anthropic,
        model,
        send,
        source,
        notes,
        buildHistory,
        evidenceHints
      );
      fullText = result.fullText;
      usage = result.usage;
    } else {
      const { client, model, baseURL } = makeOpenAIClient(apiKey);
      modelUsed = model;
      send({ type: 'status', message: `Connecting to ${new URL(baseURL).host}…` });
      debug(`model=${model} baseURL=${baseURL}`);
      const result = await runBuildWithFetchTool(
        client,
        model,
        send,
        source,
        notes,
        buildHistory,
        evidenceHints
      );
      fullText = result.fullText;
      usage = result.usage;
    }

    send({ type: 'status', message: 'Extracting collection...' });
    let collection = normalizeCollectionShape(extractCollection(fullText));

    if (!collection) {
      debug(`extract_failed fullText_length=${fullText?.length || 0}`);
      debug(`extract_preview=${String(fullText || '').slice(0, 240).replace(/\s+/g, ' ')}`);
      if (observedRequests.length) {
        const filteredObserved = filterObservedRequestsForBackend(observedRequests, source);
        if (filteredObserved.length !== observedRequests.length) {
          debug(`observed_requests_filtered=${filteredObserved.length}/${observedRequests.length}`);
        }
        const observedCollection = normalizeCollectionShape(collectionFromObservedRequests(filteredObserved));
        if (observedCollection) {
          send({
            type: 'warning',
            message:
              'No spec JSON found; built collection from filtered first-party backend-like observed calls.',
          });
          collection = observedCollection;
        }
      }
    }

    if (!collection) {
      const elapsedMs = Date.now() - buildStartedAt;
      const estimatedCostUsd = estimateCostUsd(modelUsed, usage);
      send({
        type: 'metrics',
        elapsedMs,
        usage,
        model: modelUsed,
        estimatedCostUsd,
      });
      send({
        type: 'error',
        message:
          'Insufficient verifiable API evidence to build a collection. Provide a direct OpenAPI/Swagger URL or paste Chrome DevTools "Copy as cURL" requests in Notes so the app can build from real backend traffic.',
      });
      return res.end();
    }

    if (!collection.info) collection.info = {};
    collection.info._postman_id = uuidv4();
    collection.info.schema = 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json';
    const runFolderName = withBuildRunFolder(collection);

    const name = collection.info?.name || 'API Collection';
    const endpointCount = countEndpoints(collection.item);
    if (endpointCount < 1) {
      send({
        type: 'warning',
        message:
          'No valid endpoints were extracted from this source, so Postman push was skipped. Try a direct OpenAPI/Swagger spec URL.',
      });
    }

    let postmanResult = null;
    if (postmanApiKey && endpointCount > 0) {
      send({ type: 'status', message: 'Pushing to Postman...' });
      try {
        const h = { 'Content-Type': 'application/json', 'X-Api-Key': postmanApiKey };
        const wsRes = await fetch('https://api.getpostman.com/workspaces', { headers: h });
        let workspaceId = null;
        if (wsRes.ok) {
          const ws = await wsRes.json();
          const personal = (ws.workspaces || []).find((w) => w.type === 'personal') || ws.workspaces?.[0];
          workspaceId = personal?.id;
        }
        const url = workspaceId
          ? `https://api.getpostman.com/collections?workspace=${workspaceId}`
          : 'https://api.getpostman.com/collections';
        const pr = await fetch(url, { method: 'POST', headers: h, body: JSON.stringify({ collection }) });
        const pb = await pr.json();
        if (!pr.ok) {
          const detailMsg = summarizePostmanErrorBody(pb);
          send({
            type: 'warning',
            message: `Postman (${pr.status}): ${pb.error?.message || pb.message || 'unknown error'}${
              detailMsg ? ` — ${detailMsg}` : ''
            }`,
          });
        } else {
          const uid = pb.collection?.uid || pb.collection?.id;
          postmanResult = { uid, url: uid ? `https://go.postman.co/collections/${uid}` : null };
          send({ type: 'postman_success', uid, url: postmanResult.url });
        }
      } catch (err) {
        send({ type: 'warning', message: `Postman push failed: ${err.message}` });
      }
    }

    const elapsedMs = Date.now() - buildStartedAt;
    const estimatedCostUsd = estimateCostUsd(modelUsed, usage);
    send({
      type: 'metrics',
      elapsedMs,
      usage,
      model: modelUsed,
      estimatedCostUsd,
      runFolderName,
    });
    send({ type: 'done', collection, name, endpointCount, postman: postmanResult });
    res.end();
  } catch (err) {
    logError(err);
    const elapsedMs = Date.now() - buildStartedAt;
    const estimatedCostUsd = estimateCostUsd(modelUsed, usage);
    send({
      type: 'metrics',
      elapsedMs,
      usage,
      model: modelUsed,
      estimatedCostUsd,
    });
    send({
      type: 'error',
      message: IS_PROD
        ? `Build failed: ${err?.message || 'unexpected error'}`
        : err?.stack || err?.message || 'Unexpected error',
    });
    res.end();
  }
});

// ─── /api/chat ────────────────────────────────────────────────────────────────

app.post('/api/chat', async (req, res) => {
  let message;
  try {
    message = validateMessage(req.body?.message);
  } catch (e) {
    return res.status(e.status || 400).json({ error: e.message });
  }

  const apiKey = getAiApiKey(req);
  if (!apiKey) {
    return res.status(401).json({
      error:
        'AI API key required. Add your key in the app or send the X-AI-API-Key header.',
    });
  }
  const collection = req.body?.collection;
  if (collection != null && (typeof collection !== 'object' || Array.isArray(collection))) {
    return res.status(400).json({ error: 'Invalid collection payload' });
  }

  const history = sanitizeChatHistory(req.body?.history);
  const send = sseResponder(res);

  try {
    const provider = inferAiProvider(apiKey);
    let fullText;
    if (provider === 'anthropic') {
      const anthropic = makeAnthropicClient(apiKey);
      const model = resolveAnthropicModel('chat');
      fullText = await runChatAnthropic(anthropic, model, send, collection, history, message);
    } else {
      const { client, model } = makeOpenAIClient(apiKey);

      const messages = [
        { role: 'system', content: CHAT_SYSTEM_PROMPT(collection || {}) },
        ...history.map((h) => ({ role: h.role, content: h.content })),
        { role: 'user', content: message },
      ];

      const stream = await client.chat.completions.create({
        model,
        messages,
        stream: true,
        max_tokens: AI_MAX_TOKENS_CHAT,
      });

      fullText = '';
      for await (const chunk of stream) {
        const c = chunk.choices[0]?.delta?.content;
        if (c) {
          fullText += c;
          send({ type: 'chunk', content: c });
        }
      }
    }

    const updated = extractCollection(fullText);
    if (updated?.info && updated?.item) {
      updated.info._postman_id = uuidv4();
      updated.info.schema = 'https://schema.getpostman.com/json/collection/v2.1.0/collection.json';
      send({ type: 'collection_update', collection: updated });
    }

    send({ type: 'done', text: fullText });
    res.end();
  } catch (err) {
    logError(err);
    send({
      type: 'error',
      message: IS_PROD ? 'Chat request failed. Check server logs.' : err.message,
    });
    res.end();
  }
});

// ─── /api/validate-ai-key ────────────────────────────────────────────────────

app.post('/api/validate-ai-key', validateLimiter, async (req, res) => {
  const key = req.get(AI_API_KEY_HEADER)?.trim() || req.get(LEGACY_AI_HEADER)?.trim();
  if (!key) return res.json({ valid: false, reason: 'Missing API key' });
  try {
    if (inferAiProvider(key) === 'anthropic') {
      const c = makeAnthropicClient(key);
      await c.messages.create({
        model: resolveAnthropicModel('validate'),
        max_tokens: 10,
        messages: [{ role: 'user', content: 'hi' }],
      });
    } else {
      const { client, model } = makeOpenAIClient(key);
      await client.chat.completions.create({
        model,
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 3,
      });
    }
    res.json({ valid: true });
  } catch (err) {
    const msg = err.message?.split('\n')[0] || 'Invalid key or provider error';
    res.json({ valid: false, reason: msg });
  }
});

/** @deprecated use POST /api/validate-ai-key with X-AI-API-Key */
app.post('/api/validate-anthropic-key', validateLimiter, async (req, res) => {
  const key = req.get(LEGACY_AI_HEADER)?.trim();
  if (!key) return res.json({ valid: false, reason: 'Missing API key' });
  try {
    if (inferAiProvider(key) === 'anthropic') {
      const c = makeAnthropicClient(key);
      await c.messages.create({
        model: resolveAnthropicModel('validate'),
        max_tokens: 10,
        messages: [{ role: 'user', content: 'hi' }],
      });
    } else {
      const { client, model } = makeOpenAIClient(key);
      await client.chat.completions.create({
        model,
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 3,
      });
    }
    res.json({ valid: true });
  } catch (err) {
    const msg = err.message?.split('\n')[0] || 'Invalid key or provider error';
    res.json({ valid: false, reason: msg });
  }
});

// ─── /api/validate-postman-key ────────────────────────────────────────────────

app.post('/api/validate-postman-key', validateLimiter, async (req, res) => {
  const key = req.get(POSTMAN_HEADER)?.trim();
  if (!key) return res.json({ valid: false, reason: 'Missing API key' });
  try {
    const r = await fetch('https://api.getpostman.com/me', { headers: { 'X-Api-Key': key } });
    const data = await r.json();
    if (r.ok) {
      res.json({ valid: true, user: data.user?.fullName || data.user?.username || data.user?.email });
    } else {
      res.json({ valid: false, reason: data.error?.message || data.message || `HTTP ${r.status}` });
    }
  } catch (err) {
    res.json({ valid: false, reason: err.message });
  }
});

app.use(express.static(path.join(__dirname, 'public')));

// ─── Start ────────────────────────────────────────────────────────────────────

const PORT = Number.parseInt(process.env.PORT || '3000', 10);
if (Number.isNaN(PORT) || PORT < 1 || PORT > 65535) {
  console.error('Invalid PORT');
  process.exit(1);
}

const server = app.listen(PORT, HOST, () => {
  console.log(`\nPostman Agent → http://localhost:${PORT}`);
  console.log('   Add AI and Postman keys in the web UI (stored in your browser).');
  const p = inferAiProvider('');
  if (p === 'anthropic') {
    console.log(`   Default AI target (Anthropic): ${resolveAnthropicModel('build')} @ api.anthropic.com`);
  } else {
    const { baseURL, model } = resolveAiConfig('');
    console.log(`   Default AI target (OpenAI-compatible): ${model} @ ${baseURL}`);
  }
  if (process.env.AI_BASE_URL?.trim() || process.env.AI_MODEL?.trim()) {
    console.log('   (AI_BASE_URL / AI_MODEL in .env override defaults once a key is sent)');
  }
  if (HOST === '0.0.0.0') {
    console.log(`   Listening on 0.0.0.0:${PORT} (reachable on your LAN — use only if you intend to)\n`);
  } else {
    console.log(`   Bound to ${HOST} only (local-only; not reachable from other devices)\n`);
  }
});

server.on('error', (err) => {
  if (err?.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} on ${HOST} is already in use.`);
    console.error('Stop the existing process or run with a different port, e.g. PORT=3001 npm start');
    process.exit(1);
    return;
  }
  console.error(err);
  process.exit(1);
});
