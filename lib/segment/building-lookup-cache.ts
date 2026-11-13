import { createHash } from "node:crypto"

import { getRedis, withRedisTimeout } from "@/lib/redis"
import type { OverpassElement } from "@/lib/segment/osm-buildings"

// Caches the FINISHED building lookup (merged Overpass/Nominatim elements, before ranking) so a repeat
// lookup for the same address skips Nominatim and Overpass. Ranking is always done fresh by the caller.
// A result Overpass did not answer for (`complete` false) expires after a day so the full one is
// fetched once Overpass recovers.
export const BUILDING_LOOKUP_TTL_COMPLETE_SEC = 30 * 24 * 60 * 60
export const BUILDING_LOOKUP_TTL_PARTIAL_SEC = 24 * 60 * 60
export const BUILDING_LOOKUP_MEM_MAX_ENTRIES = 200
export const BUILDING_LOOKUP_MAX_BYTES = 512 * 1024
export const BUILDING_LOOKUP_MAX_ELEMENTS = 2000
// Redis keeps the real TTL; memory entries are capped so a rehydrated entry cannot outlive Redis by weeks.
export const BUILDING_LOOKUP_MEM_TTL_MAX_SEC = 60 * 60
const KEY_PREFIX = "sv:osm-lookup:v1:"
const COORD_DECIMALS = 6

type LookupKeyInput = { lat: number; lng: number; houseNumber?: string }
type MemEntry = { expiresAt: number; elements: OverpassElement[] }

function memCache() {
  const g = globalThis as unknown as { __sunnyviewBuildingLookupCache?: Map<string, MemEntry> }
  if (!g.__sunnyviewBuildingLookupCache) g.__sunnyviewBuildingLookupCache = new Map()
  return g.__sunnyviewBuildingLookupCache
}

function lookupKey(input: LookupKeyInput) {
  const canonical = `${input.lat.toFixed(COORD_DECIMALS)},${input.lng.toFixed(COORD_DECIMALS)},${input.houseNumber ?? ""}`
  return KEY_PREFIX + createHash("sha256").update(canonical).digest("hex")
}

function ttlSeconds(complete: boolean) {
  return complete ? BUILDING_LOOKUP_TTL_COMPLETE_SEC : BUILDING_LOOKUP_TTL_PARTIAL_SEC
}

function memSet(key: string, elements: OverpassElement[], complete: boolean) {
  const mem = memCache()
  mem.delete(key)
  mem.set(key, { expiresAt: Date.now() + Math.min(ttlSeconds(complete), BUILDING_LOOKUP_MEM_TTL_MAX_SEC) * 1000, elements })
  while (mem.size > BUILDING_LOOKUP_MEM_MAX_ENTRIES) {
    const oldest = mem.keys().next().value
    if (oldest === undefined) break
    mem.delete(oldest)
  }
}

function memGet(key: string): OverpassElement[] | null {
  const mem = memCache()
  const hit = mem.get(key)
  if (!hit) return null
  if (hit.expiresAt <= Date.now()) {
    mem.delete(key)
    return null
  }
  return hit.elements
}

function isElement(v: unknown): v is OverpassElement {
  if (typeof v !== "object" || v === null) return false
  const el = v as { type?: unknown; id?: unknown }
  return typeof el.type === "string" && typeof el.id === "number" && Number.isFinite(el.id)
}

/** Redis data is untrusted: anything but a non-empty element list with a boolean `complete` is a miss. */
function parseStoredLookup(value: unknown): { elements: OverpassElement[]; complete: boolean } | null {
  if (typeof value !== "object" || value === null) return null
  const { elements, complete } = value as { elements?: unknown; complete?: unknown }
  if (typeof complete !== "boolean" || !Array.isArray(elements) || elements.length === 0 || elements.length > BUILDING_LOOKUP_MAX_ELEMENTS) return null
  return elements.every(isElement) ? { elements, complete } : null
}

async function redisGet(key: string) {
  try {
    const redis = await getRedis()
    if (!redis) return null
    return parseStoredLookup(await withRedisTimeout(() => redis.get<unknown>(key)))
  } catch {
    return null
  }
}

export async function getCachedBuildingLookup(opts: LookupKeyInput & { signal?: AbortSignal }): Promise<OverpassElement[] | null> {
  if (opts.signal?.aborted) return null
  const key = lookupKey(opts)
  const mem = memGet(key)
  if (mem) return mem
  const stored = await redisGet(key)
  if (!stored) return null
  memSet(key, stored.elements, stored.complete)
  return stored.elements
}

export async function setCachedBuildingLookup(
  opts: LookupKeyInput & { elements: OverpassElement[]; complete: boolean }
): Promise<void> {
  const { elements, complete } = opts
  if (!elements.length || Buffer.byteLength(JSON.stringify(elements), "utf8") > BUILDING_LOOKUP_MAX_BYTES) return
  const key = lookupKey(opts)
  memSet(key, elements, complete)
  try {
    const redis = await getRedis()
    if (!redis) return
    await withRedisTimeout(() => redis.set(key, { elements, complete }, { ex: ttlSeconds(complete) }))
  } catch {
    // best-effort
  }
}
