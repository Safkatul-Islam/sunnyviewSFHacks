import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { staticMapTransformFromCenter } from "@/lib/segment/mercator"
import {
  fetchOverpassBuildings,
  OVERPASS_REDIS_MAX_BYTES,
  OVERPASS_REDIS_TIMEOUT_MS,
  OVERPASS_REDIS_TTL_SEC,
  OVERPASS_COOLDOWN_MS,
  OVERPASS_HEDGE_DELAY_MS,
  OVERPASS_QUERY_BUDGET_MS,
  OVERPASS_SHORT_COOLDOWN_MS,
  pickTopOrCandidates,
  rankBuildingCandidates,
} from "@/lib/segment/osm-buildings"

const redisMock = vi.hoisted(() => ({ getRedis: vi.fn() }))
vi.mock("@/lib/redis", async () => ({
  ...(await vi.importActual<typeof import("@/lib/redis")>("@/lib/redis")),
  getRedis: redisMock.getRedis,
}))

function squareGeom(lat0: number, lon0: number, d: number) {
  return [
    { lat: lat0 - d, lon: lon0 - d },
    { lat: lat0 - d, lon: lon0 + d },
    { lat: lat0 + d, lon: lon0 + d },
    { lat: lat0 + d, lon: lon0 - d },
    { lat: lat0 - d, lon: lon0 - d },
  ]
}

describe("OSM building ranking", () => {
  it("prefers polygon containing focus point", () => {
    const tf = staticMapTransformFromCenter({
      lat: 0,
      lng: 0,
      zoom: 18,
      baseW: 520,
      baseH: 360,
      scale: 2,
      widthPx: 1040,
      heightPx: 720,
    })

    const elements: any[] = [
      { type: "way", id: 1, tags: { building: "house" }, geometry: squareGeom(0, 0, 0.00005) },
      { type: "way", id: 2, tags: { building: "house" }, geometry: squareGeom(0, 0.0003, 0.00005) },
    ]

    const scored = rankBuildingCandidates({
      elements,
      tf,
      focusPx: { x: 520, y: 360 },
      address: "123 Main St",
    })

    expect(scored[0]?.id).toBe("way:1")
    expect(scored[0]?.containsFocus).toBe(true)
  })

  it("prefers address-tag match even if not containing focus", () => {
    const tf = staticMapTransformFromCenter({
      lat: 0,
      lng: 0,
      zoom: 18,
      baseW: 520,
      baseH: 360,
      scale: 2,
      widthPx: 1040,
      heightPx: 720,
    })

    const elements: any[] = [
      { type: "way", id: 1, tags: { building: "house" }, geometry: squareGeom(0, 0, 0.00005) },
      {
        type: "way",
        id: 2,
        tags: { building: "house", "addr:housenumber": "123", "addr:street": "Main Street" },
        geometry: squareGeom(0, 0.0003, 0.00005),
      },
    ]

    const scored = rankBuildingCandidates({
      elements,
      tf,
      focusPx: { x: 520, y: 360 },
      address: "123 Main St",
    })

    expect(scored[0]?.id).toBe("way:2")
    expect(scored[0]?.addrScore).toBeGreaterThan(0)
  })

  it("returns candidates when uncertain", () => {
    const tf = staticMapTransformFromCenter({
      lat: 0,
      lng: 0,
      zoom: 18,
      baseW: 520,
      baseH: 360,
      scale: 2,
      widthPx: 1040,
      heightPx: 720,
    })

    const elements: any[] = [
      { type: "way", id: 1, tags: { building: "house" }, geometry: squareGeom(0, 0.0006, 0.00005) },
      { type: "way", id: 2, tags: { building: "house" }, geometry: squareGeom(0, -0.0006, 0.00005) },
    ]

    const scored = rankBuildingCandidates({
      elements,
      tf,
      focusPx: { x: 520, y: 360 },
      address: "123 Main St",
    })
    const pick = pickTopOrCandidates(scored)
    expect(pick.kind).toBe("candidates")
  })

  it("does not let distant addr-tags beat containment", () => {
    const tf = staticMapTransformFromCenter({
      lat: 0,
      lng: 0,
      zoom: 18,
      baseW: 520,
      baseH: 360,
      scale: 2,
      widthPx: 1040,
      heightPx: 720,
    })

    const elements: any[] = [
      // Near building contains focus.
      { type: "way", id: 1, tags: { building: "apartments" }, geometry: squareGeom(0, 0, 0.00005) },
      // Far building has perfect address tags but is hundreds of meters away.
      {
        type: "way",
        id: 2,
        tags: { building: "house", "addr:housenumber": "123", "addr:street": "Main Street" },
        geometry: squareGeom(0, 0.004, 0.00005),
      },
    ]

    const scored = rankBuildingCandidates({
      elements,
      tf,
      focusPx: { x: 520, y: 360 },
      address: "123 Main St",
    })

    expect(scored[0]?.id).toBe("way:1")
    expect(scored[0]?.containsFocus).toBe(true)
  })
})

describe("Overpass fetching", () => {
  const g = globalThis as any
  const MAIN = "https://overpass-api.de/api/interpreter"
  const COFFEE = "https://overpass.private.coffee/api/interpreter"
  const MAILRU = "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
  let fetchMock: ReturnType<typeof vi.fn>
  const savedUrl = process.env.OVERPASS_URL

  function json(body: unknown, status = 200) {
    return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })
  }

  const ok = () => json({ elements: [{ type: "way", id: 7 }] })
  const calledUrls = () => fetchMock.mock.calls.map((c) => String(c[0]))
  const query = (lat = 37.4) => fetchOverpassBuildings({ lat, lng: -122.1, radiusM: 60 })

  function hangUntilAborted(_url: string, init: RequestInit) {
    return new Promise<Response>((_resolve, reject) => {
      init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
    })
  }

  beforeEach(() => {
    delete process.env.OVERPASS_URL
    delete g.__sunnyviewOverpassCache
    delete g.__sunnyviewOverpassCooldown
    fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    redisMock.getRedis.mockReset().mockResolvedValue(null)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    if (savedUrl === undefined) delete process.env.OVERPASS_URL
    else process.env.OVERPASS_URL = savedUrl
  })

  it("POSTs the query form-encoded with identifying headers to the main instance first", async () => {
    fetchMock.mockResolvedValue(ok())
    const elements = await query()
    expect(elements).toEqual([{ type: "way", id: 7 }])
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe(MAIN)
    expect(init.method).toBe("POST")
    const headers = new Headers(init.headers)
    expect(headers.get("content-type")).toBe("application/x-www-form-urlencoded;charset=UTF-8")
    expect(headers.get("accept")).toContain("application/json")
    expect(headers.get("user-agent")).toMatch(/^sunnyviewSFHacks\/1\.0 \(\+https:\/\/github\.com\/bayshores\/sunnyviewSFHacks\)$/)
    const data = new URLSearchParams(String(init.body)).get("data")
    expect(data).toBe('[out:json][timeout:10];nwr["building"](around:60,37.4,-122.1);out tags geom;')
    expect(init.signal).toBeInstanceOf(AbortSignal)
  })

  it("uses the default order main -> private.coffee -> mail.ru and drops dead hosts", async () => {
    fetchMock.mockResolvedValue(json({}, 503))
    await expect(query()).rejects.toThrow(/Overpass failed \(503\)/)
    expect(calledUrls()).toEqual([MAIN, COFFEE, MAILRU])
    expect(calledUrls().some((u) => u.includes("kumi.systems") || u.includes("nchc.org.tw"))).toBe(false)
  })

  it("puts OVERPASS_URL entries first, ignores invalid ones and de-duplicates", async () => {
    process.env.OVERPASS_URL = " https://my.overpass.test/api/interpreter , not a url, ftp://x.test/i, " + COFFEE
    fetchMock.mockResolvedValue(json({}, 429))
    await expect(query()).rejects.toThrow(/429/)
    expect(calledUrls()).toEqual(["https://my.overpass.test/api/interpreter", COFFEE, MAIN, MAILRU])
  })

  it.each([406, 429, 500, 502, 503, 504])("falls back to the next endpoint on HTTP %i", async (status) => {
    fetchMock.mockResolvedValueOnce(json({}, status)).mockResolvedValueOnce(ok())
    expect(await query()).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN, COFFEE])
  })

  it("falls back to the next endpoint on network errors", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed")).mockResolvedValueOnce(ok())
    expect(await query()).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN, COFFEE])
  })

  it("treats an Overpass runtime-error remark as a failure and does not cache it", async () => {
    fetchMock
      .mockResolvedValueOnce(json({ elements: [], remark: "runtime error: Query timed out in \"query\" at line 1" }))
      .mockResolvedValueOnce(ok())
    expect(await query()).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN, COFFEE])
  })

  it("starts the next endpoint after the hedge delay while a slow one is still pending", async () => {
    vi.useFakeTimers()
    const signals: AbortSignal[] = []
    fetchMock.mockImplementation((url: string, init: RequestInit) => {
      signals.push(init.signal as AbortSignal)
      if (url !== MAIN) return Promise.resolve(ok())
      return new Promise((_resolve, reject) => {
        init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      })
    })
    const p = query()
    await vi.advanceTimersByTimeAsync(OVERPASS_HEDGE_DELAY_MS - 10)
    expect(calledUrls()).toEqual([MAIN])
    await vi.advanceTimersByTimeAsync(20)
    expect(await p).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN, COFFEE])
    // The losing attempt is cancelled once another endpoint answers.
    expect(signals[0]?.aborted).toBe(true)
  })

  it("does not hedge when the first endpoint answers within the hedge delay", async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(
      () => new Promise<Response>((resolve) => setTimeout(() => resolve(ok()), OVERPASS_HEDGE_DELAY_MS - 200))
    )
    const p = query()
    await vi.advanceTimersByTimeAsync(OVERPASS_QUERY_BUDGET_MS)
    expect(await p).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN])
  })

  it("lets a slow endpoint keep running for the whole budget instead of cutting it off early", async () => {
    vi.useFakeTimers()
    const answerAt = OVERPASS_QUERY_BUDGET_MS - 1_500
    fetchMock.mockImplementation((url: string, init: RequestInit) => {
      if (url === MAIN) return new Promise<Response>((resolve) => setTimeout(() => resolve(ok()), answerAt))
      return hangUntilAborted(url, init)
    })
    const p = query()
    await vi.advanceTimersByTimeAsync(answerAt + 10)
    expect(await p).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN, COFFEE, MAILRU])
  })

  it("stops the race on a non-retryable 4xx even while another endpoint is pending", async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation((url: string, init: RequestInit) => {
      if (url === COFFEE) return Promise.resolve(json({}, 400))
      return hangUntilAborted(url, init)
    })
    const p = query().catch((e: Error) => e)
    await vi.advanceTimersByTimeAsync(OVERPASS_HEDGE_DELAY_MS + 10)
    const err = await p
    expect(err).toBeInstanceOf(Error)
    expect((err as Error).message).toMatch(/Overpass failed \(400\)/)
    expect(calledUrls()).toEqual([MAIN, COFFEE])
  })

  it("keeps the whole query within its time budget", async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(
      (_url: string, init: RequestInit) =>
        new Promise((_resolve, reject) => {
          init.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
        })
    )
    const p = query()
    const settled = p.then(
      () => "resolved",
      () => "rejected"
    )
    await vi.advanceTimersByTimeAsync(OVERPASS_QUERY_BUDGET_MS + 10)
    expect(await settled).toBe("rejected")
    expect(OVERPASS_QUERY_BUDGET_MS).toBeLessThan(20_000)
  })

  it("does not retry on a non-retryable 4xx", async () => {
    fetchMock.mockResolvedValue(json({}, 400))
    await expect(query()).rejects.toThrow(/Overpass failed \(400\)/)
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("serves repeat queries from the memory cache", async () => {
    fetchMock.mockResolvedValue(ok())
    await query()
    expect(await query()).toEqual([{ type: "way", id: 7 }])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("bounds the memory cache", async () => {
    fetchMock.mockImplementation(async () => ok())
    for (let i = 0; i < 260; i++) await query(37 + i / 1000)
    expect(g.__sunnyviewOverpassCache.size).toBeLessThanOrEqual(200)
  })

  it("skips an endpoint that just failed for a cooldown period, then retries it", async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValueOnce(json({}, 429)).mockResolvedValue(ok())
    await query(37.1)
    await query(37.2)
    expect(calledUrls()).toEqual([MAIN, COFFEE, COFFEE])
    vi.advanceTimersByTime(OVERPASS_COOLDOWN_MS + 1)
    await query(37.3)
    expect(calledUrls().at(-1)).toBe(MAIN)
  })

  it("applies the operator-requested 30s cooldown after a 406", async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValueOnce(new Response("Not Acceptable", { status: 406 })).mockResolvedValue(ok())
    await query(37.1)
    vi.advanceTimersByTime(OVERPASS_COOLDOWN_MS - 1000)
    await query(37.2)
    expect(calledUrls()).toEqual([MAIN, COFFEE, COFFEE])
  })

  it.each([
    ["HTTP 503", () => json({}, 503)],
    ["a runtime-error remark", () => json({ elements: [], remark: "runtime error: Query timed out" })],
  ])("only briefly cools an endpoint down after %s", async (_label, failure) => {
    vi.useFakeTimers()
    expect(OVERPASS_SHORT_COOLDOWN_MS).toBeLessThan(OVERPASS_COOLDOWN_MS)
    fetchMock.mockResolvedValueOnce(failure()).mockResolvedValue(ok())
    await query(37.1)
    await query(37.2)
    expect(calledUrls()).toEqual([MAIN, COFFEE, COFFEE])
    vi.advanceTimersByTime(OVERPASS_SHORT_COOLDOWN_MS + 1)
    await query(37.3)
    expect(calledUrls().at(-1)).toBe(MAIN)
  })

  it("does not cool an endpoint down after a network error", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed")).mockResolvedValue(ok())
    await query(37.1)
    await query(37.2)
    expect(calledUrls()).toEqual([MAIN, COFFEE, MAIN])
  })

  it("does not cool endpoints down after timeouts, so one slow burst cannot disable Overpass", async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(hangUntilAborted)
    const first = query(37.1).catch((e: Error) => e)
    await vi.advanceTimersByTimeAsync(OVERPASS_QUERY_BUDGET_MS + 10)
    expect(await first).toBeInstanceOf(Error)
    // every endpoint was raced and ran until the query budget ran out
    expect(calledUrls()).toEqual([MAIN, COFFEE, MAILRU])

    fetchMock.mockReset()
    fetchMock.mockResolvedValue(ok())
    expect(await query(37.2)).toHaveLength(1)
    expect(calledUrls()).toEqual([MAIN])
  })

  it("respects a caller deadline shorter than the query budget", async () => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(hangUntilAborted)
    const p = fetchOverpassBuildings({ lat: 37, lng: -122, radiusM: 60, deadline: Date.now() + 2_500 }).catch((e: Error) => e)
    await vi.advanceTimersByTimeAsync(2_510)
    expect(await p).toBeInstanceOf(Error)
    expect(calledUrls()).toEqual([MAIN])
  })

  it("does not fetch when the caller deadline has (almost) passed, but still serves cache hits", async () => {
    fetchMock.mockResolvedValue(ok())
    await query(37.5)
    fetchMock.mockClear()
    const deadline = Date.now() + 100
    expect(await fetchOverpassBuildings({ lat: 37.5, lng: -122.1, radiusM: 60, deadline })).toHaveLength(1)
    await expect(fetchOverpassBuildings({ lat: 37.6, lng: -122.1, radiusM: 60, deadline })).rejects.toThrow(/budget/)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("fails fast without fetching when the caller already aborted", async () => {
    const ac = new AbortController()
    ac.abort()
    await expect(fetchOverpassBuildings({ lat: 37, lng: -122, radiusM: 60, signal: ac.signal })).rejects.toThrow()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})

describe("Overpass Redis cache", () => {
  const g = globalThis as any
  let fetchMock: ReturnType<typeof vi.fn>
  let redis: { get: ReturnType<typeof vi.fn>; set: ReturnType<typeof vi.fn> }
  const way = (id: number) => ({ type: "way", id })
  const respond = (elements: unknown[]) =>
    new Response(JSON.stringify({ elements }), { status: 200, headers: { "content-type": "application/json" } })
  const query = (lat = 37.4) => fetchOverpassBuildings({ lat, lng: -122.1, radiusM: 60 })

  beforeEach(() => {
    delete g.__sunnyviewOverpassCache
    delete g.__sunnyviewOverpassCooldown
    fetchMock = vi.fn()
    vi.stubGlobal("fetch", fetchMock)
    redis = { get: vi.fn().mockResolvedValue(null), set: vi.fn().mockResolvedValue("OK") }
    redisMock.getRedis.mockReset().mockResolvedValue(redis)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it("serves a query from Redis without calling Overpass, then from memory", async () => {
    redis.get.mockResolvedValue([way(9)])
    expect(await query()).toEqual([way(9)])
    expect(await query()).toEqual([way(9)])
    expect(fetchMock).not.toHaveBeenCalled()
    expect(redis.get).toHaveBeenCalledTimes(1)
    expect(redis.set).not.toHaveBeenCalled()
  })

  it("stores successful results in Redis for 30 days under a hashed sv:osm: key", async () => {
    fetchMock.mockResolvedValue(respond([way(7)]))
    expect(await query()).toEqual([way(7)])
    expect(OVERPASS_REDIS_TTL_SEC).toBe(30 * 24 * 60 * 60)
    expect(redis.set).toHaveBeenCalledTimes(1)
    const [key, value, opts] = redis.set.mock.calls[0]
    expect(key).toMatch(/^sv:osm:[0-9a-f]{64}$/)
    expect(value).toEqual([way(7)])
    expect(opts).toEqual({ ex: OVERPASS_REDIS_TTL_SEC })
    expect(redis.get).toHaveBeenCalledWith(key)
  })

  it("uses different keys for different queries", async () => {
    fetchMock.mockImplementation(async () => respond([way(7)]))
    await query(37.1)
    await query(37.2)
    const keys = redis.set.mock.calls.map((c) => c[0])
    expect(new Set(keys).size).toBe(2)
  })

  it("falls back to Overpass when Redis reads and writes fail", async () => {
    redis.get.mockRejectedValue(new Error("redis down"))
    redis.set.mockRejectedValue(new Error("redis down"))
    fetchMock.mockResolvedValue(respond([way(7)]))
    expect(await query()).toEqual([way(7)])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("does not wait on a slow Redis for longer than its timeout", async () => {
    vi.useFakeTimers()
    redis.get.mockImplementation(() => new Promise(() => {}))
    redis.set.mockImplementation(() => new Promise(() => {}))
    fetchMock.mockResolvedValue(respond([way(7)]))
    const p = query()
    await vi.advanceTimersByTimeAsync(OVERPASS_REDIS_TIMEOUT_MS - 50)
    expect(fetchMock).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(OVERPASS_REDIS_TIMEOUT_MS + 100)
    expect(await p).toEqual([way(7)])
    expect(OVERPASS_REDIS_TIMEOUT_MS).toBeLessThanOrEqual(500)
  })

  it("ignores malformed Redis values", async () => {
    redis.get.mockResolvedValue({ not: "an array" })
    fetchMock.mockResolvedValue(respond([way(7)]))
    expect(await query()).toEqual([way(7)])
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("does not store empty or oversized results", async () => {
    fetchMock.mockResolvedValueOnce(respond([]))
    expect(await query(37.1)).toEqual([])
    const huge = [{ type: "way", id: 1, tags: { note: "x".repeat(OVERPASS_REDIS_MAX_BYTES) } }]
    fetchMock.mockResolvedValueOnce(respond(huge))
    expect(await query(37.2)).toHaveLength(1)
    expect(redis.set).not.toHaveBeenCalled()
  })

  it("skips Redis entirely when it is not configured", async () => {
    redisMock.getRedis.mockResolvedValue(null)
    fetchMock.mockResolvedValue(respond([way(7)]))
    expect(await query()).toEqual([way(7)])
    expect(redis.get).not.toHaveBeenCalled()
  })
})
