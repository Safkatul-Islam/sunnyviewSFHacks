import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { requestClientKey, takeLocalRateLimitToken, takeRateLimitToken } from "./rate-limit"

const redisMock = vi.hoisted(() => ({ getRedis: vi.fn() }))
vi.mock("@/lib/redis", () => redisMock)

function reset() {
  delete (globalThis as unknown as { __sunnyviewRateLimitBuckets?: unknown }).__sunnyviewRateLimitBuckets
}

function bucketCount() {
  return (globalThis as unknown as { __sunnyviewRateLimitBuckets?: Map<string, unknown> }).__sunnyviewRateLimitBuckets?.size ?? 0
}

describe("takeLocalRateLimitToken", () => {
  beforeEach(() => {
    reset()
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"))
  })
  afterEach(() => {
    vi.useRealTimers()
    reset()
  })

  it("allows up to the limit then blocks with retryAfterSec", () => {
    const opts = { key: "k", limit: 2, windowMs: 60_000 }
    expect(takeLocalRateLimitToken(opts)).toEqual({ ok: true, retryAfterSec: 0 })
    expect(takeLocalRateLimitToken(opts).ok).toBe(true)
    vi.advanceTimersByTime(10_500)
    const blocked = takeLocalRateLimitToken(opts)
    expect(blocked.ok).toBe(false)
    expect(blocked.retryAfterSec).toBe(50)
  })

  it("resets after the window", () => {
    const opts = { key: "k", limit: 1, windowMs: 1000 }
    expect(takeLocalRateLimitToken(opts).ok).toBe(true)
    expect(takeLocalRateLimitToken(opts).ok).toBe(false)
    vi.advanceTimersByTime(1001)
    expect(takeLocalRateLimitToken(opts).ok).toBe(true)
  })

  it("keeps keys independent and retryAfterSec at least 1", () => {
    const a = { key: "a", limit: 1, windowMs: 100 }
    takeLocalRateLimitToken(a)
    expect(takeLocalRateLimitToken({ ...a, key: "b" }).ok).toBe(true)
    expect(takeLocalRateLimitToken(a).retryAfterSec).toBe(1)
  })

  it("sweeps expired buckets only periodically, but still honours expiry on access", () => {
    const g = globalThis as unknown as { __sunnyviewRateLimitBuckets: Map<string, unknown> }
    takeLocalRateLimitToken({ key: "old", limit: 1, windowMs: 10 })
    vi.advanceTimersByTime(31_000)
    takeLocalRateLimitToken({ key: "new", limit: 1, windowMs: 100_000 })
    expect(g.__sunnyviewRateLimitBuckets.has("old")).toBe(false)

    takeLocalRateLimitToken({ key: "short", limit: 1, windowMs: 10 })
    vi.advanceTimersByTime(20)
    // inside the sweep interval: the stale bucket stays in the map ...
    takeLocalRateLimitToken({ key: "other", limit: 1, windowMs: 100_000 })
    expect(g.__sunnyviewRateLimitBuckets.has("short")).toBe(true)
    // ... but is treated as expired when touched
    expect(takeLocalRateLimitToken({ key: "short", limit: 1, windowMs: 10 }).ok).toBe(true)
  })

  it("caps the map size and evicts least-recently-used buckets first", () => {
    const g = globalThis as unknown as { __sunnyviewRateLimitBuckets: Map<string, unknown> }
    const long = { limit: 1, windowMs: 600_000 }
    takeLocalRateLimitToken({ key: "hot", ...long })
    for (let i = 0; i < 3990; i++) takeLocalRateLimitToken({ key: `bulk:${i}`, ...long })
    takeLocalRateLimitToken({ key: "hot", ...long }) // touch (blocked) -> becomes most recent
    for (let i = 0; i < 100; i++) takeLocalRateLimitToken({ key: `flood:${i}`, ...long })
    expect(g.__sunnyviewRateLimitBuckets.size).toBeLessThanOrEqual(4000)
    expect(g.__sunnyviewRateLimitBuckets.has("hot")).toBe(true)
    expect(g.__sunnyviewRateLimitBuckets.has("bulk:0")).toBe(false)
    expect(g.__sunnyviewRateLimitBuckets.has("flood:99")).toBe(true)
  })
})

describe("takeRateLimitToken", () => {
  const opts = { key: "route:1.2.3.4", limit: 2, windowMs: 60_000 }

  beforeEach(() => {
    reset()
    redisMock.getRedis.mockReset()
    vi.spyOn(console, "warn").mockImplementation(() => {})
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.useRealTimers()
    reset()
  })

  it("uses the in-memory limiter when Redis is not configured", async () => {
    redisMock.getRedis.mockResolvedValue(null)
    expect((await takeRateLimitToken(opts)).ok).toBe(true)
    expect((await takeRateLimitToken(opts)).ok).toBe(true)
    expect(await takeRateLimitToken(opts)).toEqual({ ok: false, retryAfterSec: 60 })
  })

  it("counts in a shared Redis window keyed by route and client", async () => {
    const counts = new Map<string, number>()
    const evalFn = vi.fn(async (_script: string, keys: string[], args: unknown[]) => {
      const n = (counts.get(keys[0]!) ?? 0) + 1
      counts.set(keys[0]!, n)
      return [n, Number(args[0]) - 1500]
    })
    redisMock.getRedis.mockResolvedValue({ eval: evalFn })
    expect(await takeRateLimitToken(opts)).toEqual({ ok: true, retryAfterSec: 0 })
    expect(await takeRateLimitToken(opts)).toEqual({ ok: true, retryAfterSec: 0 })
    expect(await takeRateLimitToken(opts)).toEqual({ ok: false, retryAfterSec: 59 })
    expect(evalFn).toHaveBeenCalledWith(expect.stringContaining("INCR"), ["sv:rl:route:1.2.3.4"], [60_000])
    expect(bucketCount()).toBe(0)
  })

  it("falls back to the in-memory limiter when Redis errors or replies oddly", async () => {
    redisMock.getRedis.mockResolvedValue({ eval: vi.fn().mockRejectedValue(new Error("down")) })
    expect((await takeRateLimitToken(opts)).ok).toBe(true)
    redisMock.getRedis.mockResolvedValue({ eval: vi.fn().mockResolvedValue("OK") })
    expect((await takeRateLimitToken(opts)).ok).toBe(true)
    expect(await takeRateLimitToken(opts)).toEqual({ ok: false, retryAfterSec: 60 })
    expect(console.warn).toHaveBeenCalledTimes(1)
  })

  it("falls back when Redis is slower than the timeout", async () => {
    vi.useFakeTimers()
    redisMock.getRedis.mockResolvedValue({ eval: () => new Promise(() => {}) })
    const pending = takeRateLimitToken(opts)
    await vi.advanceTimersByTimeAsync(600)
    expect(await pending).toEqual({ ok: true, retryAfterSec: 0 })
    expect(bucketCount()).toBe(1)
  })
})

describe("requestClientKey", () => {
  const env = (e: Record<string, string>) => e as unknown as NodeJS.ProcessEnv

  it("returns unknown (one shared bucket) when no trusted header is configured, ignoring spoofable headers", () => {
    const h = new Headers({ "x-vercel-forwarded-for": "1.1.1.1", "x-real-ip": "2.2.2.2", "x-forwarded-for": "3.3.3.3" })
    expect(requestClientKey(h, env({}))).toBe("unknown")
  })

  it("defaults to x-vercel-forwarded-for on Vercel", () => {
    const h = new Headers({ "x-vercel-forwarded-for": "1.1.1.1, 9.9.9.9", "x-real-ip": "2.2.2.2" })
    expect(requestClientKey(h, env({ VERCEL: "1" }))).toBe("1.1.1.1")
  })

  it("honours only the configured TRUSTED_PROXY_HEADER", () => {
    const h = new Headers({ "x-real-ip": "2.2.2.2", "x-forwarded-for": "3.3.3.3" })
    expect(requestClientKey(h, env({ TRUSTED_PROXY_HEADER: "X-Real-IP" }))).toBe("2.2.2.2")
  })

  it("uses the right-most x-forwarded-for entry, ignoring spoofed leading values", () => {
    const h = new Headers({ "x-forwarded-for": "6.6.6.6, 3.3.3.3 , 4.4.4.4" })
    expect(requestClientKey(h, env({ TRUSTED_PROXY_HEADER: "x-forwarded-for" }))).toBe("4.4.4.4")
  })

  it("falls back to unknown when the trusted header is missing or empty, and bounds key length", () => {
    expect(requestClientKey(new Headers(), env({ TRUSTED_PROXY_HEADER: "x-real-ip" }))).toBe("unknown")
    expect(requestClientKey(new Headers({ "x-forwarded-for": " , " }), env({ TRUSTED_PROXY_HEADER: "x-forwarded-for" }))).toBe("unknown")
    const long = "a".repeat(500)
    expect(requestClientKey(new Headers({ "x-real-ip": long }), env({ TRUSTED_PROXY_HEADER: "x-real-ip" }))).toHaveLength(64)
  })
})
