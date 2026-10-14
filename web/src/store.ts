// Small localStorage helpers. Storage can be missing or full (private windows, blocked
// site data), so every read falls back to a default and every write is best effort.

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(`fletch.${key}`);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(`fletch.${key}`, JSON.stringify(value));
  } catch {
    // ignore: losing a remembered setting is fine
  }
}
