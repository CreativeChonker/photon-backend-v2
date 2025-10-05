# server_terminal.py  (threading + polling friendly)
import sys, threading, subprocess
from flask import Flask, request
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_interval=25,
    ping_timeout=60,
    logger=False,
    engineio_logger=False,
)

SESS = {}
MARKER = "__PHOTON_STDIN__:"


class Runner:
    def __init__(self, sid: str):
        self.sid = sid
        self.proc: subprocess.Popen | None = None
        self.reader: threading.Thread | None = None


def _send_stdout(sid: str, text: str):
    socketio.emit("stdout", {"text": text}, to=sid)


def _stdin_request(sid: str, prompt: str = "Input:"):
    socketio.emit("stdin_request", {"prompt": prompt}, to=sid)


def _exec_end(sid: str, status: int = 0):
    socketio.emit("exec_end", {"status": status}, to=sid)


def _reader_thread(sid: str, proc: subprocess.Popen):
    """Read child's stdout line-by-line; intercept input() marker lines."""
    try:
        while True:
            line = proc.stdout.readline()  # blocks until '\n'
            if not line:
                break
            try:
                text = line.decode(errors="replace")
            except Exception:
                text = str(line)

            # Detect our special stdin marker line
            if text.startswith(MARKER):
                prompt = text[len(MARKER):].rstrip("\n")
                _stdin_request(sid, prompt or "Input:")
                continue

            # Hide accidental REPL prompts if they appear
            if text.startswith(">>>") or text.startswith("..."):
                continue

            _send_stdout(sid, text)

        code = proc.wait()
        _exec_end(sid, code)
    except Exception as e:
        _send_stdout(sid, f"\n[reader error] {e}\n")
        try:
            code = proc.wait(timeout=0.2)
        except Exception:
            code = -1
        _exec_end(sid, code)


@socketio.on("exec_start")
def exec_start(data):
    """Start a fresh Python process and run provided code once."""
    sid = request.sid
    code = (data or {}).get("code", "")

    # Kill any previous child
    old = SESS.pop(sid, None)
    if old and old.proc and old.proc.poll() is None:
        try:
            old.proc.kill()
        except Exception:
            pass

    user_code = code if code.endswith("\n") else code + "\n"

    # IMPORTANT: keep the '\n' so the reader's readline() returns
    bootstrap = (
        "import sys, builtins\n"
        "sys.stdout.reconfigure(line_buffering=True)\n"
        f"MARKER={MARKER!r}\n"
        "def _photon_input(prompt=''):\n"
        "    # Emit one marker line with the prompt text, then newline\n"
        "    sys.stdout.write(MARKER + str(prompt) + ' ')\n"
        "    sys.stdout.write('\\n')\n"
        "    sys.stdout.flush()\n"
        "    line = sys.stdin.readline()\n"
        "    if not line:\n"
        "        return ''\n"
        "    return line.rstrip('\\n')\n"
        "builtins.input = _photon_input\n"
        + user_code
    )

    cmd = [sys.executable, "-u", "-c", bootstrap]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )

    runner = Runner(sid)
    runner.proc = proc
    SESS[sid] = runner

    runner.reader = threading.Thread(target=_reader_thread, args=(sid, proc), daemon=True)
    runner.reader.start()


@socketio.on("stdin")
def on_stdin(data):
    """Forward user input text to the child process."""
    sid = request.sid
    text = (data or {}).get("text", "")
    r = SESS.get(sid)
    if not r or not r.proc or r.proc.poll() is not None:
        return
    try:
        r.proc.stdin.write(text.encode())
        r.proc.stdin.flush()
    except Exception as e:
        _send_stdout(sid, f"\n[stdin error] {e}\n")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    r = SESS.pop(sid, None)
    if r and r.proc and r.proc.poll() is None:
        try:
            r.proc.kill()
        except Exception:
            pass


if __name__ == "__main__":
    print("🟢 Terminal server on http://127.0.0.1:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
