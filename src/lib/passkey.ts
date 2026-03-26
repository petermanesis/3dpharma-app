/**
 * Passkey-based access control — TypeScript port of auth.py
 *
 * Passkey format (before encoding): <8-char-random-id>|<duration>
 *   duration = "L" for lifetime, or a number string (hours, e.g. "2", "24")
 *
 * The raw string is base62-encoded → short, URL-safe, opaque passkey (~12 chars).
 * First-use timestamps are stored in localStorage (keyed by rand_id).
 */

const B62 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const STORE_KEY = "passkey_store";

function b62Encode(bytes: Uint8Array): string {
  let n = BigInt(0);
  for (const b of bytes) n = (n << 8n) | BigInt(b);
  if (n === 0n) return B62[0];
  let result = "";
  while (n > 0n) {
    result = B62[Number(n % 62n)] + result;
    n /= 62n;
  }
  return result;
}

function b62Decode(s: string): Uint8Array {
  let n = BigInt(0);
  for (const ch of s) {
    const idx = B62.indexOf(ch);
    if (idx === -1) throw new Error("Invalid base62 character");
    n = n * 62n + BigInt(idx);
  }
  const hex = n.toString(16).padStart(2, "0");
  const padded = hex.length % 2 ? "0" + hex : hex;
  return new Uint8Array(padded.match(/.{2}/g)!.map((b) => parseInt(b, 16)));
}

function encode(raw: string): string {
  return b62Encode(new TextEncoder().encode(raw));
}

function decode(token: string): string | null {
  try {
    return new TextDecoder().decode(b62Decode(token));
  } catch {
    return null;
  }
}

// ── Store helpers (localStorage) ─────────────────────────────────────────────

function loadStore(): Record<string, { first_used: string }> {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
  } catch {
    return {};
  }
}

function saveStore(store: Record<string, { first_used: string }>): void {
  localStorage.setItem(STORE_KEY, JSON.stringify(store));
}

// ── Public API ────────────────────────────────────────────────────────────────

export function generatePasskey(durationHours?: number): string {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let randId = "";
  for (let i = 0; i < 8; i++)
    randId += chars[Math.floor(Math.random() * chars.length)];
  const durationTag = durationHours == null ? "L" : String(Math.floor(durationHours));
  return encode(`${randId}|${durationTag}`);
}

export function validatePasskey(token: string): { valid: boolean; message: string } {
  const raw = decode(token.trim());
  if (!raw || !raw.includes("|")) return { valid: false, message: "Invalid passkey." };

  const [randId, durationTag] = raw.split("|", 2);
  const isLifetime = durationTag === "L";
  const durationHours = isLifetime ? null : parseFloat(durationTag);

  if (!isLifetime && (isNaN(durationHours!) || durationHours! <= 0))
    return { valid: false, message: "Invalid passkey." };

  const store = loadStore();

  if (!(randId in store)) {
    store[randId] = { first_used: new Date().toISOString() };
    saveStore(store);
    return { valid: true, message: "Access granted." };
  }

  if (isLifetime) return { valid: true, message: "Access granted (lifetime key)." };

  const firstUsed = new Date(store[randId].first_used);
  const elapsedHours = (Date.now() - firstUsed.getTime()) / 3_600_000;

  if (elapsedHours <= durationHours!)
    return { valid: true, message: "Access granted." };

  return {
    valid: false,
    message: `Passkey expired. It was valid for ${Math.floor(durationHours!)}h from first use (${firstUsed.toUTCString()}).`,
  };
}

export function isAuthenticated(): boolean {
  return sessionStorage.getItem("_authenticated") === "1";
}

export function setAuthenticated(): void {
  sessionStorage.setItem("_authenticated", "1");
}
