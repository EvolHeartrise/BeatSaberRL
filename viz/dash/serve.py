#!/usr/bin/env python3
"""Serve the dashboard over the laptop run archive — stdlib only.

    python3 viz/dash/serve.py                      # http://127.0.0.1:8765
    python3 viz/dash/serve.py --no-backfill        # skip the watch loop
    python3 viz/dash/serve.py --archive PATH --port 8765

Routes:
    /              viz/dash/dist/  (run `npm run build` first)
    /archive/...   ~/beat-arena-volume-backup/...  (index.json, metrics.jsonl,
                   logs, bench JSON, replays; directory listings allowed)
    /replay.html   viz/replay.html (the 3D replay viewer; drop a replay in)

Also runs `python/metrics_backfill.py --watch 30` in a subprocess so
metrics.jsonl + index.json follow the archive while a campaign's rsync
loop fills it. Read-only over the archive; never touches the pod.
"""
import argparse
import http.server
import os
import posixpath
import subprocess
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
REPLAY = os.path.join(os.path.dirname(HERE), "replay.html")
BACKFILL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "python",
                        "metrics_backfill.py")


def make_handler(archive):
    class H(http.server.SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
            path = posixpath.normpath(path)
            if path == "/replay.html":
                return REPLAY
            if path.startswith("/archive"):
                rest = path[len("/archive"):].lstrip("/")
                return os.path.join(archive, *[p for p in rest.split("/") if p])
            rest = path.lstrip("/")
            return os.path.join(DIST, *[p for p in rest.split("/") if p])

        # ETag = (mtime_ns, size): the dashboard's poll sends If-None-Match
        # and gets a body-less 304 while a file is unchanged (it also
        # compares content, so a plain http.server without this works too)
        def send_head(self):
            self._etag = None
            path = self.translate_path(self.path)
            try:
                st = os.stat(path)
            except OSError:
                st = None
            if st is not None and os.path.isfile(path):
                self._etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
                inm = self.headers.get("If-None-Match")
                if inm and self._etag in [t.strip() for t in inm.split(",")]:
                    self.send_response(304)
                    self.end_headers()
                    return None
            return super().send_head()

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            if getattr(self, "_etag", None):
                self.send_header("ETag", self._etag)
            super().end_headers()

        def log_message(self, fmt, *args):
            first = str(args[0]) if args else ""
            if "index.json" in first or "metrics" in first:
                return  # the 15 s poll is noise
            super().log_message(fmt, *args)

    return H


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        # a browser or curl hanging up mid-transfer is not an error
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive",
                    default=os.path.expanduser("~/beat-arena-volume-backup"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--no-backfill", action="store_true")
    ap.add_argument("--watch", type=float, default=30.0,
                    help="backfill re-scan interval (s)")
    a = ap.parse_args()
    if not os.path.isdir(a.archive):
        sys.exit(f"archive not found: {a.archive}")
    if not os.path.exists(os.path.join(DIST, "index.html")):
        print(f"! {DIST}/index.html missing — run `npm run build` in "
              f"{HERE} (the /archive route still works)", flush=True)
    proc = None
    if not a.no_backfill:
        proc = subprocess.Popen([sys.executable, BACKFILL, "--archive",
                                 a.archive, "--watch", str(a.watch), "-q"])
        print(f"[serve] backfill watch every {a.watch:g}s (pid {proc.pid})",
              flush=True)
    srv = Server((a.bind, a.port), make_handler(a.archive))
    print(f"[serve] http://{a.bind}:{a.port}/  (archive {a.archive})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
