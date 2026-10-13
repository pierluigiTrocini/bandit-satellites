"""Server di simulazione live: esegue un episodio BSK-RL e lo trasmette al viewer via SSE.

Solo libreria standard. Endpoint:
  GET  /stream    Server-Sent Events: `snapshot` (traccia v2 accumulata), poi un evento `round`
                  per ogni round (delta + stato) e `status` a fine episodio
  GET  /status    stato corrente (JSON)
  GET  /trace.json  traccia accumulata finora
  POST /control   {"action": "pause" | "resume" | "restart" (con "seed" opzionale)}
                  {"action": "speed", "speed": <secondi simulati per secondo reale> | "max"}

Velocita': 1 = tempo reale, 60 = un minuto simulato al secondo, "max" = senza pause (il piu'
veloce possibile). Il round [t0, t1] viene inviato quando il tempo simulato raggiunge t0, quindi il
viewer ha sempre davanti i dati del round che sta mostrando. Il cambio di velocita' vale subito.

Uso: uv run python scripts/sim_server.py [--sats 3] [--hours 24] [--speed 60]
(--hours 0 = episodio senza limite di tempo: la storia in memoria cresce con la durata)
"""

import argparse
import json
import queue
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from constellation import Sim


class Hub:
    def __init__(self, sim, seed, speed, paused):
        self.sim, self.seed, self.speed, self.paused = sim, seed, speed, paused
        self.cond = threading.Condition(threading.RLock())
        self.subs, self.restart_seed = [], None
        sim.reset(seed)

    def status(self):
        return dict(paused=self.paused, speed="max" if self.speed is None else self.speed, done=self.sim.done,
                    t=self.sim.t, seed=self.seed)

    def broadcast(self, event, payload):
        text = f"event: {event}\ndata: {payload}\n\n"
        for q in list(self.subs):
            try:
                q.put_nowait(text)
            except queue.Full:            # client troppo lento: lo si scollega (si ricollega da solo)
                self.subs.remove(q)

    def subscribe(self):
        q = queue.Queue(maxsize=20000)
        with self.cond:
            q.put(f"event: snapshot\ndata: {json.dumps(dict(trace=self.sim.trace, status=self.status()))}\n\n")
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.cond:
            if q in self.subs:
                self.subs.remove(q)

    def control(self, msg):
        with self.cond:
            act = msg.get("action")
            if act == "pause":
                self.paused = True
            elif act == "resume":
                self.paused = False
            elif act == "speed":
                v = msg.get("speed")
                if v == "max":
                    self.speed = None
                elif isinstance(v, (int, float)) and v > 0:
                    self.speed = float(v)
                else:
                    raise ValueError("speed deve essere un numero > 0 oppure 'max'")
            elif act == "restart":
                self.restart_seed = int(msg.get("seed", self.seed))
            else:
                raise ValueError(f"azione sconosciuta: {act!r}")
            self.broadcast("status", json.dumps(self.status()))
            self.cond.notify_all()
            return self.status()

    def run(self):
        anchor = None                      # (istante reale, istante simulato, velocita') per il ritmo
        while True:
            with self.cond:
                if self.restart_seed is not None:
                    self.seed, self.restart_seed = self.restart_seed, None
                    self.sim.reset(self.seed)
                    self.broadcast("snapshot", json.dumps(dict(trace=self.sim.trace, status=self.status())))
                    anchor = None
                    continue
                if self.paused or self.sim.done:
                    anchor = None
                    self.cond.wait(0.5)
                    continue
                speed, t = self.speed, self.sim.t
                if anchor is None or anchor[2] != speed:
                    anchor = (time.monotonic(), t, speed)
                if speed is not None:
                    wait = anchor[0] + (t - anchor[1]) / speed - time.monotonic()
                    if wait > 0:
                        self.cond.wait(min(wait, 0.2))   # si sveglia anche per pause/velocita'/restart
                        continue
                delta = self.sim.step()
                delta["status"] = self.status()
                self.broadcast("round", json.dumps(delta, separators=(",", ":")))
            if speed is None:
                time.sleep(0.001)          # a velocita' massima il lock va lasciato ai thread HTTP (non e' equo)


def make_handler(hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, code, body, ctype="application/json"):
            body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            for k, v in (("Access-Control-Allow-Origin", "*"), ("Access-Control-Allow-Headers", "Content-Type"),
                         ("Access-Control-Allow-Methods", "GET, POST, OPTIONS")):
                self.send_header(k, v)
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/status"):
                return self._send(200, json.dumps(hub.status()))
            if self.path.startswith("/trace.json"):
                with hub.cond:
                    return self._send(200, json.dumps(hub.sim.trace))
            if not self.path.startswith("/stream"):
                return self._send(404, '{"error":"not found"}')
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            q = hub.subscribe()
            try:
                while True:
                    try:
                        self.wfile.write(q.get(timeout=15).encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(q)

        def do_POST(self):
            if not self.path.startswith("/control"):
                return self._send(404, '{"error":"not found"}')
            try:
                msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._send(200, json.dumps(hub.control(msg)))
            except (ValueError, TypeError) as e:
                self._send(400, json.dumps({"error": str(e)}))

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--sats", type=int, default=3)
    ap.add_argument("--planes", type=int, default=None)
    ap.add_argument("--altitude", type=float, default=500.0)
    ap.add_argument("--inc", type=float, default=55.0)
    ap.add_argument("--targets", type=int, default=600)
    ap.add_argument("--round", type=float, default=60.0)
    ap.add_argument("--track-dt", type=float, default=10.0)
    ap.add_argument("--hours", type=float, default=24.0, help="durata episodio [h simulate]; 0 = infinita")
    ap.add_argument("--speed", default="60", help="secondi simulati per secondo reale, o 'max'")
    ap.add_argument("--paused", action="store_true", help="parte in pausa")
    a = ap.parse_args()

    sim = Sim(sats=a.sats, planes=a.planes, altitude=a.altitude, inc=a.inc, targets=a.targets,
              round_s=a.round, track_dt=a.track_dt, seconds=a.hours * 3600)
    hub = Hub(sim, a.seed, None if a.speed == "max" else float(a.speed), a.paused)

    def loop():
        try:
            hub.run()
        except Exception:
            traceback.print_exc()
            with hub.cond:
                hub.sim.done = True
                hub.broadcast("status", json.dumps(dict(hub.status(), error="la simulazione si e' interrotta")))

    threading.Thread(target=loop, daemon=True).start()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(hub))
    server.daemon_threads = True
    print(f"sim_server: {a.sats} satelliti, seed {a.seed}, velocita' {a.speed}, http://{a.host}:{a.port}/stream", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
