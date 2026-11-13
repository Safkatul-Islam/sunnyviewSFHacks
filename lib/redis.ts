type AnyRedis = {
  get: <T = unknown>(key: string) => Promise<T | null>
  set: (key: string, value: unknown, opts?: { ex?: number }) => Promise<unknown>
  eval: (script: string, keys: string[], args: unknown[]) => Promise<unknown>
}

let redisPromise: Promise<AnyRedis | null> | null = null

export async function getRedis(): Promise<AnyRedis | null> {
  // KV_REST_API_* is the naming used by Vercel's Upstash/KV integrations.
  const url = (process.env.UPSTASH_REDIS_REST_URL || process.env.KV_REST_API_URL)?.trim()
  const token = (process.env.UPSTASH_REDIS_REST_TOKEN || process.env.KV_REST_API_TOKEN)?.trim()
  if (!url || !token) return null

  if (!redisPromise) {
    redisPromise = (async () => {
      try {
        // Optional dependency. If not installed, this import will fail and Redis caching is disabled.
        // @ts-ignore
        const mod = (await import("@upstash/redis")) as any
        const Redis = mod?.Redis
        if (!Redis) return null
        return new Redis({ url, token }) as AnyRedis
      } catch {
        // Optional dependency: if not installed, just disable Redis caching.
        return null
      }
    })()
  }

  return redisPromise
}

export const REDIS_TIMEOUT_MS = 500

/**
 * Runs a best-effort Redis call: resolves null when it takes longer than REDIS_TIMEOUT_MS or fails,
 * so a slow or broken Redis never blocks or breaks the caller.
 */
export async function withRedisTimeout<T>(run: () => Promise<T>): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), REDIS_TIMEOUT_MS)
  })
  try {
    const pending = run()
    pending.catch(() => {}) // a late failure after the timeout must not become an unhandled rejection
    return await Promise.race([pending, timeout])
  } catch {
    return null
  } finally {
    clearTimeout(timer)
  }
}
