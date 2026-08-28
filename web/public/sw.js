/* take5bot service worker.
 *
 * Exists for one reason: a cold visit otherwise re-downloads the ~3.6 MB
 * champion weights (net-attn.t5n) and the ~119 kB WASM engine every
 * time, which is brutal on cellular.
 *
 * Hand-rolled rather than vite-plugin-pwa: the plugin's default
 * maximumFileSizeToCacheInBytes is 2 MB, which would SILENTLY skip the
 * very file this exists to cache. Runtime caching sidesteps that, and
 * the whole policy fits on one screen.
 *
 * Caching policy
 *   - navigations           network-first, fall back to the cached shell
 *   - /assets/* (hashed)    cache-first, immutable by construction
 *   - *.wasm, *.t5n         cache-first
 *   - everything else       passthrough
 *
 * NOTE: net-attn.t5n is served from public/ and is therefore NOT
 * content-hashed. Bump CACHE_VERSION whenever the weights change, or
 * clients will keep serving the old net.
 */

const CACHE_VERSION = "take5bot-v1";
const BASE = new URL("./", self.location).pathname;
const SHELL = `${BASE}index.html`;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_VERSION)
      .then((cache) => cache.add(new Request(SHELL, { cache: "reload" })))
      .catch(() => {})
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

/** Big, effectively-immutable payloads worth holding on to. */
function isPrecious(url) {
  return (
    url.pathname.startsWith(`${BASE}assets/`) ||
    url.pathname.endsWith(".wasm") ||
    url.pathname.endsWith(".t5n")
  );
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_VERSION);
  const hit = await cache.match(request);
  if (hit) return hit;
  const res = await fetch(request);
  // Only cache complete, successful, same-origin responses; a 206 from a
  // range request would poison the cache with a partial body.
  if (res.ok && res.status === 200) cache.put(request, res.clone());
  return res;
}

async function networkFirst(request) {
  const cache = await caches.open(CACHE_VERSION);
  try {
    const res = await fetch(request);
    if (res.ok) cache.put(SHELL, res.clone());
    return res;
  } catch (err) {
    const hit = (await cache.match(SHELL)) || (await cache.match(request));
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === "navigate") {
    event.respondWith(networkFirst(request));
    return;
  }
  if (isPrecious(url)) {
    event.respondWith(cacheFirst(request));
  }
});
