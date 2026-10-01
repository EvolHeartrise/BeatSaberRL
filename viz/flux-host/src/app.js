// flux.flee.systems: the Flux WebXR game behind a Discord login.
//
// The game is the single-file viewer build (viz/replay.html), bundled INTO
// this Worker (src/index.js) — it is never a public static asset, so the
// only way a browser receives its bytes is `GET /` with a valid session
// cookie whose Discord user id is on the whitelist (checked on every
// request, so removing an id locks that player out immediately). Everyone
// else gets a small login / "not whitelisted" page and nothing of the game.
//
// Flow: GET /login -> Discord OAuth2 (scope `identify`, CSRF `state` held
// in a short-lived cookie) -> GET /callback exchanges the code, reads
// /users/@me, sets an HMAC-signed session cookie -> GET / serves the game.
//
// Config (wrangler secrets): DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET,
// SESSION_SECRET (random, >= 32 chars), ALLOWED_DISCORD_IDS (comma /
// whitespace separated Discord user ids); var PUBLIC_ORIGIN (wrangler.jsonc;
// the OAuth redirect URI is PUBLIC_ORIGIN + "/callback", registered with the
// Discord app). Missing config fails closed.
//
// `__Host-` cookies: Secure, Path=/, no Domain — no other *.flee.systems host
// can plant or overwrite them (login CSRF / session fixation).

const SESSION_COOKIE = "__Host-flux_session";
const STATE_COOKIE = "__Host-flux_oauth_state";
const SESSION_TTL_S = 7 * 24 * 3600;
const STATE_TTL_S = 600;
const DISCORD_AUTHORIZE = "https://discord.com/oauth2/authorize";
const DISCORD_TOKEN = "https://discord.com/api/oauth2/token";
const DISCORD_ME = "https://discord.com/api/users/@me";

const enc = new TextEncoder();

// ---- base64url + HMAC-signed tokens ----------------------------------------
function b64url(bytes) {
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function unb64url(s) {
  const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

async function hmacKey(secret) {
  return crypto.subtle.importKey("raw", enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function sign(payload, secret) {
  const body = b64url(enc.encode(JSON.stringify(payload)));
  const sig = await crypto.subtle.sign("HMAC", await hmacKey(secret), enc.encode(body));
  return `${body}.${b64url(new Uint8Array(sig))}`;
}

// The payload if the signature is valid and it has not expired, else null.
export async function verify(token, secret, now = Date.now() / 1000) {
  if (typeof token !== "string") return null;
  const dot = token.indexOf(".");
  if (dot <= 0 || dot !== token.lastIndexOf(".")) return null;
  const body = token.slice(0, dot);
  try {
    // crypto.subtle.verify compares in constant time.
    const ok = await crypto.subtle.verify("HMAC", await hmacKey(secret),
      unb64url(token.slice(dot + 1)), enc.encode(body));
    if (!ok) return null;
    const p = JSON.parse(new TextDecoder().decode(unb64url(body)));
    return typeof p.exp === "number" && p.exp > now ? p : null;
  } catch {
    return null;
  }
}

// ---- config ----------------------------------------------------------------
export function allowedIds(env) {
  const ids = new Set();
  for (const s of String(env.ALLOWED_DISCORD_IDS ?? "").split(/[\s,]+/)) {
    if (/^\d{5,25}$/.test(s)) ids.add(s);
    else if (s) console.warn(`ALLOWED_DISCORD_IDS: ignoring ${JSON.stringify(s)} (not a Discord user id)`);
  }
  return ids;
}

function configured(env) {
  return Boolean(env.DISCORD_CLIENT_ID && env.DISCORD_CLIENT_SECRET
    && env.SESSION_SECRET && String(env.SESSION_SECRET).length >= 32
    && /^https?:\/\/[^/]+$/.test(env.PUBLIC_ORIGIN ?? ""));
}

// ---- cookies -----------------------------------------------------------------
function cookies(request) {
  const out = {};
  for (const part of (request.headers.get("Cookie") ?? "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = part.slice(i + 1).trim();
  }
  return out;
}

function setCookie(name, value, maxAge) {
  return `${name}=${value}; Path=/; Max-Age=${maxAge}; HttpOnly; Secure; SameSite=Lax`;
}

// ---- responses ---------------------------------------------------------------
const BASE_HEADERS = {
  "Cache-Control": "private, no-store",
  "X-Content-Type-Options": "nosniff",
  "X-Frame-Options": "DENY",
  "Referrer-Policy": "no-referrer",
};

function html(body, status = 200, extra = {}) {
  return new Response(body, {
    status, headers: { ...BASE_HEADERS, "Content-Type": "text/html; charset=utf-8", ...extra },
  });
}

function redirect(location, setCookies = []) {
  const h = new Headers({ ...BASE_HEADERS, Location: location });
  for (const c of setCookies) h.append("Set-Cookie", c);
  return new Response(null, { status: 302, headers: h });
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function page(title, inner, status = 200, extra = {}) {
  return html(`<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex">
<title>Flux</title><style>
:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;
background:#0b0d14;color:#e8e8f0;font:16px/1.5 system-ui,sans-serif}
main{max-width:26rem;padding:2rem 1.5rem;text-align:center}h1{font-size:1.6rem;margin:0 0 .5rem}
p{color:#a8acc0}a.btn,button{display:inline-block;margin-top:1rem;padding:.7rem 1.4rem;border:0;
border-radius:.5rem;background:#5865f2;color:#fff;font:inherit;text-decoration:none;cursor:pointer}
button.ghost{background:#2a2e3d}code{background:#1c1f2b;padding:.1rem .35rem;border-radius:.25rem}
</style></head><body><main>${title ? `<h1>${esc(title)}</h1>` : ""}${inner}</main></body></html>`, status, extra);
}

const logoutForm = `<form method="post" action="/logout"><button class="ghost">log out</button></form>`;

// ---- the handler -------------------------------------------------------------
// `gameHtml` is the game build; `fetchImpl` is injectable for tests.
export async function handle(request, env, gameHtml, fetchImpl = fetch) {
  const url = new URL(request.url);
  const method = request.method;
  // Must match the redirect URI registered with the Discord app exactly;
  // fixed by config, never taken from the request's Host.
  const callback = `${env.PUBLIC_ORIGIN}/callback`;

  if (!configured(env)) {
    return page("Flux", "<p>This site is not configured yet.</p>", 503);
  }

  if (url.pathname === "/" && (method === "GET" || method === "HEAD")) {
    const s = await verify(cookies(request)[SESSION_COOKIE], env.SESSION_SECRET);
    if (!s) {
      return page("", `<a class="btn" href="/login">Log in with Discord</a>`, 401);
    }
    if (!allowedIds(env).has(s.sub)) {
      return page("Not on the list", `<p>You're logged in as <b>${esc(s.name)}</b>,
but this Discord account isn't whitelisted.</p>
<p>Your Discord id: <code>${esc(s.sub)}</code></p>${logoutForm}`, 403);
    }
    return html(method === "HEAD" ? null : gameHtml, 200, {
      // WebXR in the game page itself; nothing else needs powerful features.
      "Permissions-Policy": "xr-spatial-tracking=(self)",
    });
  }

  if (url.pathname === "/login" && method === "GET") {
    const state = b64url(crypto.getRandomValues(new Uint8Array(24)));
    const q = new URLSearchParams({
      response_type: "code",
      client_id: env.DISCORD_CLIENT_ID,
      scope: "identify",
      redirect_uri: callback,
      state,
      prompt: "none",
    });
    return redirect(`${DISCORD_AUTHORIZE}?${q}`, [setCookie(STATE_COOKIE, state, STATE_TTL_S)]);
  }

  if (url.pathname === "/callback" && method === "GET") {
    const clearState = setCookie(STATE_COOKIE, "", 0);
    const state = url.searchParams.get("state");
    const code = url.searchParams.get("code");
    const expected = cookies(request)[STATE_COOKIE];
    if (!code || !state || !expected || state !== expected) {
      return page("Login failed", `<p>The login link expired or was invalid.</p>
<a class="btn" href="/login">Try again</a>`, 400, { "Set-Cookie": clearState });
    }
    let me = null;
    try {
      const tokenRes = await fetchImpl(DISCORD_TOKEN, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          grant_type: "authorization_code",
          code,
          redirect_uri: callback,
          client_id: env.DISCORD_CLIENT_ID,
          client_secret: env.DISCORD_CLIENT_SECRET,
        }),
      });
      const token = tokenRes.ok ? await tokenRes.json() : null;
      const meRes = token?.access_token
        ? await fetchImpl(DISCORD_ME, { headers: { Authorization: `Bearer ${token.access_token}` } })
        : null;
      me = meRes?.ok ? await meRes.json() : null;
    } catch (e) {
      // Network error or a non-JSON 200 (e.g. a challenge page).
      console.warn(`Discord OAuth exchange failed: ${e}`);
    }
    if (!me || typeof me.id !== "string" || !/^\d+$/.test(me.id)) {
      return page("Login failed", `<p>Discord didn't confirm the login.</p>
<a class="btn" href="/login">Try again</a>`, 502, { "Set-Cookie": clearState });
    }
    const session = await sign({
      sub: me.id,
      name: String(me.global_name || me.username || me.id).slice(0, 64),
      exp: Math.floor(Date.now() / 1000) + SESSION_TTL_S,
    }, env.SESSION_SECRET);
    return redirect("/", [clearState, setCookie(SESSION_COOKIE, session, SESSION_TTL_S)]);
  }

  if (url.pathname === "/logout" && method === "POST") {
    return redirect("/", [setCookie(SESSION_COOKIE, "", 0)]);
  }

  return new Response("Not found", { status: 404, headers: BASE_HEADERS });
}
