import { createHash } from "node:crypto"

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import {
  BUILDING_LOOKUP_MAX_ELEMENTS,
  BUILDING_LOOKUP_MEM_MAX_ENTRIES,
  BUILDING_LOOKUP_MEM_TTL_MAX_SEC,
  BUILDING_LOOKUP_TTL_COMPLETE_SEC,
  BUILDING_LOOKUP_TTL_PARTIAL_SEC,
  getCachedBuildingLookup,
  setCachedBuildingLookup,
} from "@/lib/segment/building-lookup-cache"

const redisMock = vi.hoisted(() => ({ getRedis: vi.fn() }))
vi.mock("@/lib/redis", async () => ({
  ...(await vi.importActual<typeof import("@/lib/redis")>("@/lib/redis")),
  getRedis: redisMock.getRedis,
}))

const g = globalThis as any
const ELEMENTS = [{ type: "way", id: 1, tags: { building: "house" } }]
const BASE = { lat: 37.4, lng: -122.1, houseNumber: "12" }

function redisKey(lat: string, lng: string, houseNumber: string) {
  return "sv:osm-lookup:v1:" + createHash("sha256").update(`${lat},${lng},${houseNumber}`).digest("hex")
}

describe("segment/building-lookup-cache", () => {
  let redis: { get: ReturnType<typeof vi.fn>; set: ReturnType<typeof vi.fn> }

  beforeEach(() => {
    delete g.__sunnyviewBuildingLookupCache
    redis = { get: vi.fn().mockResolvedValue(null), set: vi.fn().mockResolvedValue("OK") }
    redisMock.getRedis.mockReset().mockResolvedValue(null)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("round-trips through memory without Redis", async () => {
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
    await setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })
    expect(await getCachedBuildingLookup({ ...BASE })).toEqual(ELEMENTS)
  })

  it("ignores float noise beyond 6 decimals and separates house numbers", async () => {
    await setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })
    expect(await getCachedBuildingLookup({ ...BASE, lat: 37.4 + 1e-9, lng: -122.1 - 1e-9 })).toEqual(ELEMENTS)
    expect(await getCachedBuildingLookup({ ...BASE, houseNumber: "14" })).toBeNull()
    expect(await getCachedBuildingLookup({ ...BASE, houseNumber: undefined })).toBeNull()
    expect(await getCachedBuildingLookup({ ...BASE, lat: 37.4001 })).toBeNull()
  })

  it("writes to Redis under the hashed key with a 30 day or 1 day expiry", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    await setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })
    await setCachedBuildingLookup({ ...BASE, houseNumber: undefined, elements: ELEMENTS, complete: false })
    expect(BUILDING_LOOKUP_TTL_COMPLETE_SEC).toBe(30 * 24 * 60 * 60)
    expect(BUILDING_LOOKUP_TTL_PARTIAL_SEC).toBe(24 * 60 * 60)
    expect(redis.set).toHaveBeenNthCalledWith(
      1,
      redisKey("37.400000", "-122.100000", "12"),
      { elements: ELEMENTS, complete: true },
      { ex: 30 * 24 * 60 * 60 }
    )
    expect(redis.set).toHaveBeenNthCalledWith(
      2,
      redisKey("37.400000", "-122.100000", ""),
      { elements: ELEMENTS, complete: false },
      { ex: 24 * 60 * 60 }
    )
  })

  it("does not store empty or oversized results", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    await setCachedBuildingLookup({ ...BASE, elements: [], complete: true })
    const huge = [{ type: "way", id: 2, tags: { note: "x".repeat(512 * 1024) } }]
    await setCachedBuildingLookup({ ...BASE, elements: huge, complete: true })
    expect(redis.set).not.toHaveBeenCalled()
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
  })

  it("returns a Redis hit when memory is empty and fills memory", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    redis.get.mockResolvedValueOnce({ elements: ELEMENTS, complete: true })
    expect(await getCachedBuildingLookup({ ...BASE })).toEqual(ELEMENTS)
    expect(redis.get).toHaveBeenCalledWith(redisKey("37.400000", "-122.100000", "12"))
    redisMock.getRedis.mockResolvedValue(null)
    expect(await getCachedBuildingLookup({ ...BASE })).toEqual(ELEMENTS)
  })

  it.each([
    ["a string", "nope"],
    ["a bare array", ELEMENTS],
    ["empty elements", { elements: [], complete: true }],
    ["missing complete", { elements: ELEMENTS }],
    ["non-string type", { elements: [{ type: 3, id: 1 }], complete: true }],
    ["non-finite id", { elements: [{ type: "way", id: "1" }], complete: true }],
    ["null element", { elements: [null], complete: true }],
  ])("treats a malformed Redis value (%s) as a miss", async (_name, value) => {
    redisMock.getRedis.mockResolvedValue(redis)
    redis.get.mockResolvedValue(value)
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
  })

  it("returns null when Redis get hangs past the timeout", async () => {
    vi.useFakeTimers()
    redisMock.getRedis.mockResolvedValue(redis)
    redis.get.mockReturnValue(new Promise(() => {}))
    const p = getCachedBuildingLookup({ ...BASE })
    await vi.advanceTimersByTimeAsync(600)
    expect(await p).toBeNull()
  })

  it("swallows Redis errors on get, set and getRedis", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    redis.get.mockRejectedValue(new Error("boom"))
    redis.set.mockRejectedValue(new Error("boom"))
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
    await expect(setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })).resolves.toBeUndefined()
    redisMock.getRedis.mockRejectedValue(new Error("no redis"))
    expect(await getCachedBuildingLookup({ ...BASE, houseNumber: "99" })).toBeNull()
    await expect(
      setCachedBuildingLookup({ ...BASE, houseNumber: "99", elements: ELEMENTS, complete: true })
    ).resolves.toBeUndefined()
  })

  it("misses when the signal is already aborted", async () => {
    await setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })
    redisMock.getRedis.mockClear()
    const ac = new AbortController()
    ac.abort()
    expect(await getCachedBuildingLookup({ ...BASE, signal: ac.signal })).toBeNull()
    expect(redisMock.getRedis).not.toHaveBeenCalled()
  })

  it("evicts the oldest entry beyond the memory cap", async () => {
    for (let i = 0; i <= BUILDING_LOOKUP_MEM_MAX_ENTRIES; i++) {
      await setCachedBuildingLookup({ ...BASE, lat: 10 + i * 0.001, elements: ELEMENTS, complete: true })
    }
    expect(g.__sunnyviewBuildingLookupCache.size).toBe(BUILDING_LOOKUP_MEM_MAX_ENTRIES)
    expect(await getCachedBuildingLookup({ ...BASE, lat: 10 })).toBeNull()
    expect(await getCachedBuildingLookup({ ...BASE, lat: 10.001 })).toEqual(ELEMENTS)
  })

it("caps the memory life of a set entry at 1 hour even when complete", async () => {
    vi.useFakeTimers()
    expect(BUILDING_LOOKUP_MEM_TTL_MAX_SEC).toBe(3600)
    await setCachedBuildingLookup({ ...BASE, elements: ELEMENTS, complete: true })
    await vi.advanceTimersByTimeAsync((BUILDING_LOOKUP_MEM_TTL_MAX_SEC - 1) * 1000)
    expect(await getCachedBuildingLookup({ ...BASE })).toEqual(ELEMENTS)
    await vi.advanceTimersByTimeAsync(2000)
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
  })

  it("caps the memory life of a Redis-rehydrated entry at 1 hour", async () => {
    vi.useFakeTimers()
    redisMock.getRedis.mockResolvedValue(redis)
    redis.get.mockResolvedValueOnce({ elements: ELEMENTS, complete: true })
    expect(await getCachedBuildingLookup({ ...BASE })).toEqual(ELEMENTS)
    redisMock.getRedis.mockResolvedValue(null)
    await vi.advanceTimersByTimeAsync((BUILDING_LOOKUP_MEM_TTL_MAX_SEC + 1) * 1000)
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
  })

  it("measures the size cap in bytes, not characters", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    const text = "é".repeat(300 * 1024)
    const multiByte = [{ type: "way", id: 3, tags: { note: text } }]
    const json = JSON.stringify(multiByte)
    expect(json.length).toBeLessThan(512 * 1024)
    expect(Buffer.byteLength(json, "utf8")).toBeGreaterThan(512 * 1024)
    await setCachedBuildingLookup({ ...BASE, elements: multiByte, complete: true })
    expect(redis.set).not.toHaveBeenCalled()
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
  })

  it("treats a stored list with too many elements as a miss", async () => {
    redisMock.getRedis.mockResolvedValue(redis)
    const many = Array.from({ length: BUILDING_LOOKUP_MAX_ELEMENTS + 1 }, (_v, i) => ({ type: "way", id: i }))
    redis.get.mockResolvedValue({ elements: many, complete: true })
    expect(BUILDING_LOOKUP_MAX_ELEMENTS).toBe(2000)
    expect(await getCachedBuildingLookup({ ...BASE })).toBeNull()
    redis.get.mockResolvedValue({ elements: many.slice(0, BUILDING_LOOKUP_MAX_ELEMENTS), complete: true })
    expect(await getCachedBuildingLookup({ ...BASE })).toHaveLength(BUILDING_LOOKUP_MAX_ELEMENTS)
  })
})
