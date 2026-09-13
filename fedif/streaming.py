"""Private local IPC for adaptive ICS inputs to the existing MP-SPDZ VM."""

from contextlib import suppress
import queue
import subprocess
import threading
import time

from .ics import LocalCountTables


def run_ics_online(argv, cwd, input_file, Z, y, spec, party_id, timeout):
    """Stream tables over this party's stdin; monitor only public stdout events.

    A dedicated reader drains output even while stdin blocks, avoiding pipe
    deadlocks on large inputs. The watchdog covers writes as well as reads.
    No local table, dummy point, label or feature is written to an output log.
    """
    start = time.monotonic()
    events = queue.Queue()
    tables = LocalCountTables(Z, y, spec, party_id)
    table_seconds = 0.0
    expired = threading.Event()
    process = None
    reader = watchdog = None
    with (cwd / "online.log").open("w", encoding="utf-8") as log, \
            (cwd / "online.stderr.log").open("w", encoding="utf-8") as errors:
        try:
            process = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors,
                                       text=True, encoding="utf-8", errors="replace",
                                       bufsize=1, shell=False)

            def drain():
                try:
                    for line in process.stdout:
                        log.write(line)
                        if line.startswith("FEDIF_"):
                            events.put(line.strip())
                except Exception as exc:
                    events.put(exc)
                finally:
                    events.put(None)

            def abort():
                expired.set()
                with suppress(OSError):
                    process.kill()

            reader = threading.Thread(target=drain, name="fedif-public-output", daemon=True)
            watchdog = threading.Timer(timeout, abort)
            watchdog.daemon = True
            reader.start()
            watchdog.start()
            # The initial input layout is identical to CountSamples mode.
            with input_file.open("r", encoding="ascii") as initial:
                for chunk in iter(lambda: initial.read(65536), ""):
                    process.stdin.write(chunk)
            process.stdin.flush()

            def send(values):
                flat = values.reshape(-1)
                for begin in range(0, len(flat), 4096):
                    process.stdin.write(" ".join(str(int(x)) for x in flat[begin:begin + 4096]) + "\n")
                process.stdin.flush()

            ended = False
            while True:
                remaining = timeout - (time.monotonic() - start)
                if remaining <= 0 or expired.is_set():
                    raise TimeoutError("ICS online phase timed out")
                event = events.get(timeout=remaining)
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event
                fields = event.split()
                if fields[0] == "FEDIF_ICS_QUERY":
                    if ended or len(fields) != 4:
                        raise ValueError("Malformed ICS request")
                    before = time.monotonic()
                    points, counts = tables.query(*map(int, fields[1:]))
                    table_seconds += time.monotonic() - before
                    if tables.size:  # publicly empty parties have no table inputs
                        send(points)
                        send(counts)
                elif fields[0] == "FEDIF_SPLIT":
                    if ended or len(fields) != 5:
                        raise ValueError("Malformed public split")
                    tables.record_split(*map(int, fields[1:]))
                elif fields[0] == "FEDIF_END":
                    if ended:
                        raise ValueError("Duplicate ICS end marker")
                    tables.finish()
                    ended = True
                    process.stdin.close()
            code = process.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
            if expired.is_set():
                raise TimeoutError("ICS online phase timed out")
            if code or not ended:
                raise RuntimeError("Incomplete or failed MPC process")
        except Exception as exc:
            raise RuntimeError(f"MP-SPDZ ICS online phase failed; inspect {cwd / 'online.log'} "
                               f"and {cwd / 'online.stderr.log'}. Use a fresh session_id.") from exc
        finally:
            if watchdog is not None:
                watchdog.cancel()
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait()
                with suppress(OSError):
                    process.stdin.close()
                if reader is not None:
                    reader.join(timeout=5)
                process.stdout.close()
    return {"online_seconds": time.monotonic() - start,
            "ics_local_table_seconds": table_seconds, "ics_queries": tables.queries}
