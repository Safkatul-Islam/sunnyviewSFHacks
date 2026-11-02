// With Upstash Redis configured (see lib/redis.ts), buckets are shared fixed-window
// counters in Redis, so limits hold across serverless instances. Without Redis, or
// when it errors or is slow, each instance falls back to its own in-memory buckets
// (state on globalThis), which makes limits best-effort on multi-instance deployments.
//
// Client identity (see requestClientKey) is only as trustworthy as the proxy in
// front of the app: forwarding headers are client-controlled unless a trusted
// proxy overwrites them. We therefore honour one explicitly configured header.

import { getRedis } from "@/lib/redis"

type RateBucket = {
  count: number
  resetAt: number
}

type RateLimitResult = {
  ok: boolean
  retryAfterSec: number
}

type RateLimitOptions = { key: string; limit: number; windowMs: number }

const MAX_BUCKETS = 4000
const SWEEP_INTERVAL_MS = 30_000
const MAX_KEY_LENGTH = 64
const REDIS_KEY_PREFIX = "sv:rl:"
const REDIS_TIMEOUT_MS = 500
const REDIS_WARN_INTERVAL_MS = 60_000

// Atomic fixed window: count the hit, start the window on the first hit, return [count, ttl ms].
const REDIS_WINDOW_SCRIPT = `
local count = redis.call("INCR", KEYS[1])
local ttl = redis.call("PTTL", KEYS[1])
if ttl < 0 then
  redis.call("PEXPIRE", KEYS[1], ARGV[1])
  ttl = tonumber(ARGV[1])
end
return {count, ttl}
`

let lastSweepAt = 0
let lastRedisWarnAt = 0

function bucketMap() {
  const g = globalThis as unknown as { __sunnyviewRateLimitBuckets?: Map<string, RateBucket> }
  if (!g.__sunnyviewRateLimitBuckets) g.__sunnyviewRateLimitBuckets = new Map()
  return g.__sunnyviewRateLimitBuckets
}

/** Drops expired buckets, at most once per SWEEP_INTERVAL_MS (not on every call). */
function sweepExpired(now: number) {
  if (now >= lastSweepAt && now - lastSweepAt < SWEEP_INTERVAL_MS) return
  lastSweepAt = now
  const buckets = bucketMap()
  for (const [key, value] of buckets.entries()) {
    if (value.resetAt <= now) buckets.delete(key)
  }
}

/** Map iteration order is insertion order and touched keys are re-inserted, so the head is least recently used. */
function evictLeastRecentlyUsed() {
  const buckets = bucketMap()
  while (buckets.size > MAX_BUCKETS) {
    const oldestKey = buckets.keys().next().value
    if (oldestKey === undefined) break
    buckets.delete(oldestKey)
  }
}

/** Per-instance in-memory limiter; also the fallback when Redis is unavailable. */
export function takeLocalRateLimitToken(opts: RateLimitOptions): RateLimitResult {
  const now = Date.now()
  sweepExpired(now)

  const buckets = bucketMap()
  const existing = buckets.get(opts.key)
  const active = !existing || existing.resetAt <= now ? { count: 0, resetAt: now + opts.windowMs } : existing

  const blocked = active.count >= opts.limit
  if (!blocked) active.count += 1

  buckets.delete(opts.key)
  buckets.set(opts.key, active) // re-insert so recently used keys are evicted last
  evictLeastRecentlyUsed()

  if (blocked) {
    return { ok: false, retryAfterSec: Math.max(1, Math.ceil((active.resetAt - now) / 1000)) }
  }
  return { ok: true, retryAfterSec: 0 }
}

function warnRedisFallback(error: unknown) {
  const now = Date.now()
  if (now - lastRedisWarnAt < REDIS_WARN_INTERVAL_MS) return
  lastRedisWarnAt = now
  console.warn("[rate-limit] redis unavailable, using in-memory limiter", {
    message: error instanceof Error ? error.message : String(error),
  })
}

async function evalWithTimeout(run: () => Promise<unknown>): Promise<unknown> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timed out after ${REDIS_TIMEOUT_MS}ms`)), REDIS_TIMEOUT_MS)
  })
  const pending = run()
  pending.catch(() => {}) // a late failure after the timeout must not become an unhandled rejection
  try {
    return await Promise.race([pending, timeout])
  } finally {
    clearTimeout(timer)
  }
}

function parseWindowReply(reply: unknown): { count: number; ttlMs: number } | null {
  if (!Array.isArray(reply) || reply.length < 2) return null
  const count = Number(reply[0])
  const ttlMs = Number(reply[1])
  return Number.isFinite(count) && Number.isFinite(ttlMs) ? { count, ttlMs } : null
}

/** Shared limiter: Redis fixed window when configured, otherwise the in-memory limiter. */
export async function takeRateLimitToken(opts: RateLimitOptions): Promise<RateLimitResult> {
  const redis = await getRedis()
  if (!redis) return takeLocalRateLimitToken(opts)

  try {
    const reply = await evalWithTimeout(() =>
      redis.eval(REDIS_WINDOW_SCRIPT, [`${REDIS_KEY_PREFIX}${opts.key}`], [opts.windowMs])
    )
    const counter = parseWindowReply(reply)
    if (!counter) throw new Error("unexpected redis reply")
    if (counter.count <= opts.limit) return { ok: true, retryAfterSec: 0 }
    return { ok: false, retryAfterSec: Math.max(1, Math.ceil(counter.ttlMs / 1000)) }
  } catch (error) {
    warnRedisFallback(error)
    return takeLocalRateLimitToken(opts)
  }
}

/**
 * Derive a client identifier from ONE trusted header.
 * - On Vercel (VERCEL=1) the default is `x-vercel-forwarded-for`, which the
 *   platform overwrites.
 * - Elsewhere only the header named by TRUSTED_PROXY_HEADER is read (for example
 *   `x-real-ip`, or `x-forwarded-for` where the right-most entry, appended by the
 *   nearest proxy, is used).
 * - With nothing configured we return "unknown": every client shares one bucket.
 *   That is deliberately conservative (a spoofable per-IP key would let anyone
 *   bypass limits by rotating header values), at the cost of a shared limit.
 */
export function requestClientKey(headers: Headers, env: NodeJS.ProcessEnv = process.env): string {
  const configured = env.TRUSTED_PROXY_HEADER?.trim().toLowerCase()
  const headerName = configured || (env.VERCEL === "1" ? "x-vercel-forwarded-for" : "")
  if (!headerName) return "unknown"

  const parts = (headers.get(headerName) ?? "")
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean)
  const picked = headerName === "x-forwarded-for" ? parts[parts.length - 1] : parts[0]
  return picked ? picked.slice(0, MAX_KEY_LENGTH) : "unknown"
}
