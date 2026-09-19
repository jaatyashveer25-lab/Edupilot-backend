'use strict';

/**
 * POST /api/chat  (Vercel serverless function)
 *
 * Request
 *   Headers: Authorization: Bearer <Supabase user access token>
 *   Body   : {
 *              "user_id":   "<uuid>",
 *              "device_id": "<string>",
 *              "message":   "<string>",
 *              "history":   [{ "role": "user" | "assistant", "content": "<string>" }]  // optional
 *            }
 *
 * Flow
 *   1. Validate input
 *   2. Verify the Supabase JWT belongs to user_id            (401 / 403)
 *   3. Rate limit: 20 requests / minute / user_id            (429)
 *   4. Confirm an active row exists in user_sessions
 *      for (user_id, device_id)                              (403)
 *   5. Call Gemini with GEMINI_API_KEY; on HTTP 429 retry once
 *      with GEMINI_BACKUP_KEY
 *
 * Required env vars
 *   GEMINI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY
 * Optional env vars
 *   GEMINI_BACKUP_KEY          fallback key used when the primary returns 429
 *   GEMINI_MODEL               default: gemini-2.5-flash
 *   ALLOWED_ORIGINS            comma-separated list, default "*"
 *   UPSTASH_REDIS_REST_URL     } shared rate-limit store (recommended in production;
 *   UPSTASH_REDIS_REST_TOKEN   } without it, limits are enforced per warm instance only)
 */

const { createClient } = require('@supabase/supabase-js');

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const GEMINI_ENDPOINT = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;
const GEMINI_TIMEOUT_MS = 25_000;

const RATE_LIMIT_MAX = 20;
const RATE_LIMIT_WINDOW_MS = 60_000;

const MAX_MESSAGE_CHARS = 4_000;
const MAX_HISTORY_ITEMS = 20;
const MAX_DEVICE_ID_CHARS = 200;

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const SYSTEM_PROMPT =
  'You are a friendly, accurate study assistant inside an EdTech app. ' +
  'Explain concepts clearly and step by step, use simple examples, and adapt to the ' +
  "student's level. Encourage learning rather than just handing over answers. " +
  'If you are unsure, say so instead of guessing.';

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
function setCors(req, res) {
  const allowed = (process.env.ALLOWED_ORIGINS || '*')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  const origin = req.headers.origin;

  if (allowed.includes('*')) {
    res.setHeader('Access-Control-Allow-Origin', '*');
  } else if (origin && allowed.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  }
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  res.setHeader('Access-Control-Max-Age', '86400');
}

function readBody(req) {
  let body = req.body;
  if (Buffer.isBuffer(body)) body = body.toString('utf8');
  if (typeof body === 'string') {
    try {
      body = JSON.parse(body);
    } catch {
      return null;
    }
  }
  return body && typeof body === 'object' && !Array.isArray(body) ? body : null;
}

function getBearerToken(req) {
  const match = /^Bearer\s+(.+)$/i.exec(req.headers.authorization || '');
  return match ? match[1].trim() : null;
}

/** Returns { error } or { userId, deviceId, message, history }. */
function validateInput(body) {
  if (!body) return { error: 'Request body must be valid JSON.' };

  const userId = typeof body.user_id === 'string' ? body.user_id.trim() : '';
  const deviceId = typeof body.device_id === 'string' ? body.device_id.trim() : '';
  const message = typeof body.message === 'string' ? body.message.trim() : '';

  if (!UUID_RE.test(userId)) return { error: 'user_id must be a valid UUID.' };
  if (!deviceId || deviceId.length > MAX_DEVICE_ID_CHARS) {
    return { error: 'device_id is required.' };
  }
  if (!message) return { error: 'message is required.' };
  if (message.length > MAX_MESSAGE_CHARS) {
    return { error: `message must be at most ${MAX_MESSAGE_CHARS} characters.` };
  }

  let history = [];
  if (body.history !== undefined) {
    if (!Array.isArray(body.history)) return { error: 'history must be an array.' };
    history = body.history
      .filter(
        (h) =>
          h &&
          (h.role === 'user' || h.role === 'assistant' || h.role === 'model') &&
          typeof h.content === 'string' &&
          h.content.trim()
      )
      .slice(-MAX_HISTORY_ITEMS)
      .map((h) => ({
        role: h.role === 'user' ? 'user' : 'model',
        text: h.content.trim().slice(0, MAX_MESSAGE_CHARS),
      }));
    // Gemini conversations should start with a user turn.
    while (history.length && history[0].role !== 'user') history.shift();
  }

  return { userId, deviceId, message, history };
}

// ----------------------------------------------------------------------------
// Rate limiting (fixed window, 20 requests / minute / user_id)
// ----------------------------------------------------------------------------
const memoryBuckets = new Map();

function memoryRateLimit(userId) {
  const now = Date.now();

  if (memoryBuckets.size > 5_000) {
    for (const [key, bucket] of memoryBuckets) {
      if (bucket.resetAt <= now) memoryBuckets.delete(key);
    }
  }

  let bucket = memoryBuckets.get(userId);
  if (!bucket || bucket.resetAt <= now) {
    bucket = { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    memoryBuckets.set(userId, bucket);
  }
  bucket.count += 1;

  return {
    allowed: bucket.count <= RATE_LIMIT_MAX,
    retryAfterSec: Math.max(1, Math.ceil((bucket.resetAt - now) / 1000)),
  };
}

async function redisRateLimit(userId) {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  const now = Date.now();
  const windowId = Math.floor(now / RATE_LIMIT_WINDOW_MS);
  const key = `rl:chat:${userId}:${windowId}`;

  const response = await fetch(`${url.replace(/\/$/, '')}/pipeline`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
    body: JSON.stringify([
      ['INCR', key],
      ['EXPIRE', key, 120],
    ]),
    signal: AbortSignal.timeout(3_000),
  });
  if (!response.ok) throw new Error(`Redis responded ${response.status}`);

  const data = await response.json();
  const count = Number(data?.[0]?.result);
  if (!Number.isFinite(count)) throw new Error('Unexpected Redis response');

  return {
    allowed: count <= RATE_LIMIT_MAX,
    retryAfterSec: Math.max(1, Math.ceil(((windowId + 1) * RATE_LIMIT_WINDOW_MS - now) / 1000)),
  };
}

async function checkRateLimit(userId) {
  if (process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN) {
    try {
      return await redisRateLimit(userId);
    } catch (err) {
      console.error('[chat] Redis rate limiter failed, using in-memory fallback:', err.message);
    }
  }
  return memoryRateLimit(userId);
}

// ----------------------------------------------------------------------------
// Gemini
// ----------------------------------------------------------------------------
async function callGemini(apiKey, payload) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), GEMINI_TIMEOUT_MS);
  try {
    const response = await fetch(GEMINI_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-goog-api-key': apiKey, // header, not query string, so the key never lands in logs
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json().catch(() => null);
    return {
      status: response.status,
      ok: response.ok,
      data,
      retryAfter: response.headers.get('retry-after'),
    };
  } finally {
    clearTimeout(timer);
  }
}

function extractReply(data) {
  const parts = data?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) return '';
  return parts.map((p) => (typeof p.text === 'string' ? p.text : '')).join('').trim();
}

// ----------------------------------------------------------------------------
// Handler
// ----------------------------------------------------------------------------
module.exports = async function handler(req, res) {
  setCors(req, res);
  res.setHeader('Cache-Control', 'no-store');

  if (req.method === 'OPTIONS') return res.status(204).end();
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST, OPTIONS');
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  const { GEMINI_API_KEY, GEMINI_BACKUP_KEY, SUPABASE_URL, SUPABASE_ANON_KEY } = process.env;
  if (!GEMINI_API_KEY || !SUPABASE_URL || !SUPABASE_ANON_KEY) {
    console.error('[chat] Missing required environment variables.');
    return res.status(500).json({ error: 'Server is not configured correctly.' });
  }

  // 1. Validate input -------------------------------------------------------
  const input = validateInput(readBody(req));
  if (input.error) return res.status(400).json({ error: input.error });
  const { userId, deviceId, message, history } = input;

  const token = getBearerToken(req);
  if (!token) return res.status(401).json({ error: 'Missing Authorization bearer token.' });

  try {
    // 2. Verify the caller's identity ---------------------------------------
    // The anon-key client forwards the user's JWT, so Row Level Security
    // (auth.uid() = user_id) applies to every query below.
    const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
      auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
      global: { headers: { Authorization: `Bearer ${token}` } },
    });

    const { data: authData, error: authError } = await supabase.auth.getUser(token);
    if (authError || !authData?.user) {
      return res.status(401).json({ error: 'Invalid or expired token.' });
    }
    if (authData.user.id !== userId) {
      return res.status(403).json({ error: 'user_id does not match the authenticated user.' });
    }

    // 3. Rate limit (after auth so nobody can burn another user's quota) -----
    const limit = await checkRateLimit(userId);
    if (!limit.allowed) {
      res.setHeader('Retry-After', String(limit.retryAfterSec));
      return res.status(429).json({
        error: 'Too many requests. Please wait a moment and try again.',
        retry_after_seconds: limit.retryAfterSec,
      });
    }

    // 4. Verify an active session for this device ----------------------------
    const { data: session, error: sessionError } = await supabase
      .from('user_sessions')
      .select('id')
      .eq('user_id', userId)
      .eq('device_id', deviceId)
      .eq('is_active', true)
      .limit(1)
      .maybeSingle();

    if (sessionError) {
      console.error('[chat] Session lookup failed:', sessionError.message);
      return res.status(500).json({ error: 'Could not verify session.' });
    }
    if (!session) {
      return res.status(403).json({ error: 'No active session for this device. Please log in again.' });
    }

    // 5. Call Gemini, falling back to the backup key on 429 ------------------
    const payload = {
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents: [
        ...history.map((h) => ({ role: h.role, parts: [{ text: h.text }] })),
        { role: 'user', parts: [{ text: message }] },
      ],
      generationConfig: { temperature: 0.7, maxOutputTokens: 2048 },
    };

    let result = await callGemini(GEMINI_API_KEY, payload);

    if (result.status === 429 && GEMINI_BACKUP_KEY) {
      console.warn('[chat] Primary Gemini key rate-limited (429); retrying with backup key.');
      result = await callGemini(GEMINI_BACKUP_KEY, payload);
    }

    if (result.status === 429) {
      const retryAfter = Number.parseInt(result.retryAfter, 10) || 30;
      res.setHeader('Retry-After', String(retryAfter));
      return res.status(429).json({
        error: 'The AI service is busy right now. Please try again shortly.',
        retry_after_seconds: retryAfter,
      });
    }

    if (!result.ok) {
      console.error('[chat] Gemini error:', result.status, result.data?.error?.message || '');
      return res.status(502).json({ error: 'The AI service is unavailable. Please try again.' });
    }

    const reply = extractReply(result.data);
    if (!reply) {
      const blocked = result.data?.promptFeedback?.blockReason;
      if (blocked) {
        return res.status(422).json({ error: 'That request could not be answered. Try rephrasing it.' });
      }
      return res.status(502).json({ error: 'The AI service returned an empty response.' });
    }

    return res.status(200).json({ reply });
  } catch (err) {
    if (err?.name === 'AbortError') {
      return res.status(504).json({ error: 'The AI service took too long to respond.' });
    }
    console.error('[chat] Unexpected error:', err?.message || err);
    return res.status(500).json({ error: 'Internal server error.' });
  }
};
