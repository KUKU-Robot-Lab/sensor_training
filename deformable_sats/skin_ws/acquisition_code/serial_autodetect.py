"""Best-effort serial port auto-detection.

DUE/loadcell ports move around whenever a cable lands in a different USB
slot. This matches connected ports by VID:PID first (most reliable), then by
a description keyword, so the hardcoded COM numbers in final_logger.py /
loadcell_bin_logger.py only need to be a fallback, not the source of truth.
"""

import sys

try:
    from serial.tools import list_ports
except ImportError:
    list_ports = None


def _score_port(port, vid_pid_set, keywords):
    if (port.vid, port.pid) in vid_pid_set:
        return 2
    haystack = f"{port.description} {port.hwid}".lower()
    if any(kw in haystack for kw in keywords):
        return 1
    return 0


def find_candidates(vid_pid_set=(), keywords=()):
    """Ports matching vid_pid_set or keywords, best match first."""
    if list_ports is None:
        return []
    scored = [
        (port, _score_port(port, vid_pid_set, keywords))
        for port in list_ports.comports()
    ]
    matches = [(p, s) for p, s in scored if s > 0]
    matches.sort(key=lambda ps: ps[1], reverse=True)
    return [p for p, _ in matches]


def resolve_port(label, explicit_port, env_port, default_port, vid_pid_set=(), keywords=()):
    """Pick a serial port for `label` (e.g. "DUE", "Loadcell").

    Priority: explicit_port (CLI arg) > env_port (env var) > a single
    auto-detected match > default_port (hardcoded fallback, with a warning).
    """
    if explicit_port:
        print(f"{label}: using explicitly configured port {explicit_port}.", file=sys.stderr)
        return explicit_port
    if env_port:
        print(f"{label}: using port {env_port} from environment override.", file=sys.stderr)
        return env_port

    candidates = find_candidates(vid_pid_set, keywords)
    if len(candidates) == 1:
        port = candidates[0]
        print(f"{label}: auto-detected {port.device} ({port.description}).", file=sys.stderr)
        return port.device
    if len(candidates) > 1:
        listing = ", ".join(f"{p.device} ({p.description})" for p in candidates)
        print(
            f"{label}: multiple candidate ports found ({listing}); "
            f"falling back to default {default_port}. Pass an explicit port to disambiguate.",
            file=sys.stderr,
        )
        return default_port

    print(
        f"{label}: no port auto-detected; falling back to default {default_port}.",
        file=sys.stderr,
    )
    return default_port
