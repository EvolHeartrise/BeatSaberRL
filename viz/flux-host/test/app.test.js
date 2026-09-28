// Node tests for the flux.flee.systems gate (src/app.js): the game bytes
// reach a browser only with a valid, unexpired, correctly-signed session
// whose Discord id is whitelisted; the OAuth flow (Discord mocked) checks
// its CSRF state; missing config fails closed.
// Run: node test/app.test.js
import assert from "node:assert/strict";
import { allowedIds, handle, sign, verify } from "../src/app.js";

const GAME = "<!doctype html><title>FLUX-GAME-MARKER</title>";
const SECRET = "s".repeat(40);
const ENV = {
  DISCORD_CLIENT_ID: "123",
  DISCORD_CLIENT_SECRET: "shh",
  SESSION_SECRET: SECRET,
  ALLOWED_DISCORD_IDS: "111111111111111111, 222222222222222222\n333333333333333333",
};
const ORIGIN = "https://flux.flee.systems";
const now = () => Math.floor(Date.now() / 1000);

const req = (path, { method = "GET", cookie } = {}) =>
  new Request(ORIGIN + path, { method, headers: cookie ? { Cookie: cookie } : {} });
const session = (sub, extra = {}) =>
  sign({ sub, name: "tester", exp: now() + 3600, ...extra }, SECRET);
const body = async (res) => (res.body ? await res.text() : "");
const noGame = async (res, why) => assert.ok(!(await body(res)).includes("FLUX-GAME-MARKER"), why);
const noFetch = () => { throw new Error("unexpected fetch"); };

// ---- tokens ---------------------------------------------------------------
{
  const t = await sign({ sub: "1", exp: now() + 60 }, SECRET);
  assert.equal((await verify(t, SECRET)).sub, "1");
  assert.equal(await verify(t, "x".repeat(40)), null, "wrong key");
  assert.equal(await verify(t + "A", SECRET), null, "tampered sig");
  const [b, s] = t.split(".");
  const forged = btoa(JSON.stringify({ sub: "2", exp: now() + 60 })).replace(/=+$/, "");
  assert.equal(await verify(`${forged}.${s}`, SECRET), null, "swapped payload");
  assert.equal(await verify(`${b}.${s}.x`, SECRET), null);
  assert.equal(await verify(await sign({ sub: "1", exp: now() - 1 }, SECRET), SECRET), null, "expired");
  assert.equal(await verify(await sign({ sub: "1" }, SECRET), SECRET), null, "no exp");
  for (const junk of [undefined, "", ".", "a.b", "%%%.%%%"]) assert.equal(await verify(junk, SECRET), null);
}

// ---- whitelist parsing ------------------------------------------------------
assert.deepEqual([...allowedIds(ENV)],
  ["111111111111111111", "222222222222222222", "333333333333333333"]);
assert.equal(allowedIds({}).size, 0);
assert.equal(allowedIds({ ALLOWED_DISCORD_IDS: "abc, 12" }).size, 0);

// ---- the gate ---------------------------------------------------------------
{
  const ok = await handle(req("/", { cookie: `flux_session=${await session("222222222222222222")}` }), ENV, GAME, noFetch);
  assert.equal(ok.status, 200);
  assert.equal(await ok.text(), GAME);
  assert.match(ok.headers.get("Cache-Control"), /no-store/);
  assert.match(ok.headers.get("Cache-Control"), /private/);

  const head = await handle(req("/", { method: "HEAD", cookie: `flux_session=${await session("222222222222222222")}` }), ENV, GAME, noFetch);
  assert.equal(head.status, 200);

  const cases = [
    ["no cookie", undefined, 401],
    ["garbage cookie", "flux_session=garbage", 401],
    ["not whitelisted", `flux_session=${await session("444444444444444444")}`, 403],
    ["expired", `flux_session=${await session("111111111111111111", { exp: now() - 5 })}`, 401],
    ["signed with another key", `flux_session=${await sign({ sub: "111111111111111111", exp: now() + 60 }, "k".repeat(40))}`, 401],
  ];
  for (const [why, cookie, status] of cases) {
    const res = await handle(req("/", { cookie }), ENV, GAME, noFetch);
    assert.equal(res.status, status, why);
    await noGame(res, why);
  }

  // Revocation: dropping an id from the list locks out an existing session.
  const cookie = `flux_session=${await session("111111111111111111")}`;
  const revoked = await handle(req("/", { cookie }), { ...ENV, ALLOWED_DISCORD_IDS: "222222222222222222" }, GAME, noFetch);
  assert.equal(revoked.status, 403);
  await noGame(revoked, "revoked");

  // Missing config fails closed, even with a would-be-valid session.
  for (const k of ["DISCORD_CLIENT_ID", "DISCORD_CLIENT_SECRET", "SESSION_SECRET"]) {
    const res = await handle(req("/", { cookie }), { ...ENV, [k]: undefined }, GAME, noFetch);
    assert.equal(res.status, 503, k);
    await noGame(res, k);
  }
  const short = await handle(req("/", { cookie }), { ...ENV, SESSION_SECRET: "short" }, GAME, noFetch);
  assert.equal(short.status, 503);

  // Only `/` ever serves the game.
  for (const p of ["/index.html", "/replay.html", "/game", "//", "/?x=1".replace("?", "%3F")]) {
    const res = await handle(req(p, { cookie }), ENV, GAME, noFetch);
    assert.notEqual(res.status, 200, p);
    await noGame(res, p);
  }
  const post = await handle(req("/", { method: "POST", cookie }), ENV, GAME, noFetch);
  assert.equal(post.status, 404);
  await noGame(post, "POST /");
}

// ---- OAuth flow (Discord mocked) --------------------------------------------
function discord({ id = "111111111111111111", tokenOk = true, meOk = true } = {}) {
  const calls = [];
  const impl = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    if (String(url).endsWith("/oauth2/token")) {
      const p = new URLSearchParams(init.body);
      assert.equal(p.get("code"), "the-code");
      assert.equal(p.get("client_secret"), "shh");
      assert.equal(p.get("redirect_uri"), `${ORIGIN}/callback`);
      return tokenOk ? Response.json({ access_token: "at" }) : new Response("no", { status: 400 });
    }
    if (String(url).endsWith("/users/@me")) {
      assert.equal(init.headers.Authorization, "Bearer at");
      return meOk ? Response.json({ id, username: "u<script>", global_name: null }) : new Response("no", { status: 401 });
    }
    throw new Error(`unexpected ${url}`);
  };
  return { impl, calls };
}

const cookieOf = (res, name) => {
  for (const c of res.headers.getSetCookie()) {
    if (c.startsWith(`${name}=`)) return c.split(";")[0].slice(name.length + 1);
  }
  return undefined;
};

{
  const login = await handle(req("/login"), ENV, GAME, noFetch);
  assert.equal(login.status, 302);
  const loc = new URL(login.headers.get("Location"));
  assert.equal(loc.origin + loc.pathname, "https://discord.com/oauth2/authorize");
  assert.equal(loc.searchParams.get("scope"), "identify");
  assert.equal(loc.searchParams.get("client_id"), "123");
  assert.equal(loc.searchParams.get("redirect_uri"), `${ORIGIN}/callback`);
  const state = loc.searchParams.get("state");
  assert.ok(state.length >= 20);
  assert.equal(cookieOf(login, "flux_oauth_state"), state);
  for (const c of login.headers.getSetCookie()) assert.match(c, /HttpOnly; Secure; SameSite=Lax/);

  // Happy path: whitelisted user gets a session, which then opens the game.
  const d = discord();
  const cb = await handle(req(`/callback?code=the-code&state=${state}`, { cookie: `flux_oauth_state=${state}` }), ENV, GAME, d.impl);
  assert.equal(cb.status, 302);
  assert.equal(cb.headers.get("Location"), "/");
  assert.equal(d.calls.length, 2);
  const s = cookieOf(cb, "flux_session");
  assert.equal(cookieOf(cb, "flux_oauth_state"), "", "state cookie cleared");
  assert.equal((await verify(s, SECRET)).sub, "111111111111111111");
  const game = await handle(req("/", { cookie: `flux_session=${s}` }), ENV, GAME, noFetch);
  assert.equal(await game.text(), GAME);

  // A non-whitelisted Discord user can log in but still never gets the game;
  // their (escaped) name and id are shown so they can ask to be added.
  const d2 = discord({ id: "555555555555555555" });
  const cb2 = await handle(req(`/callback?code=the-code&state=${state}`, { cookie: `flux_oauth_state=${state}` }), ENV, GAME, d2.impl);
  const denied = await handle(req("/", { cookie: `flux_session=${cookieOf(cb2, "flux_session")}` }), ENV, GAME, noFetch);
  assert.equal(denied.status, 403);
  const deniedText = await denied.text();
  assert.ok(!deniedText.includes("FLUX-GAME-MARKER"));
  assert.ok(deniedText.includes("555555555555555555"));
  assert.ok(deniedText.includes("u&lt;script&gt;") && !deniedText.includes("u<script>"), "name escaped");

  // CSRF: state missing / mismatched / no state cookie -> no Discord call, no session.
  for (const [q, cookie] of [
    ["code=the-code", `flux_oauth_state=${state}`],
    [`code=the-code&state=${state}`, undefined],
    [`code=the-code&state=${state}`, "flux_oauth_state=other"],
    [`state=${state}`, `flux_oauth_state=${state}`],
  ]) {
    const res = await handle(req(`/callback?${q}`, { cookie }), ENV, GAME, noFetch);
    assert.equal(res.status, 400, q);
    assert.equal(cookieOf(res, "flux_session"), undefined);
  }

  // Discord rejects the code or the token -> no session.
  for (const opts of [{ tokenOk: false }, { meOk: false }]) {
    const res = await handle(req(`/callback?code=the-code&state=${state}`, { cookie: `flux_oauth_state=${state}` }), ENV, GAME, discord(opts).impl);
    assert.equal(res.status, 502);
    assert.equal(cookieOf(res, "flux_session"), undefined);
  }

  // Logout clears the session (POST only).
  const out = await handle(req("/logout", { method: "POST" }), ENV, GAME, noFetch);
  assert.equal(out.status, 302);
  assert.equal(cookieOf(out, "flux_session"), "");
  assert.equal((await handle(req("/logout"), ENV, GAME, noFetch)).status, 404);
}

console.log("flux-host: all tests passed");
