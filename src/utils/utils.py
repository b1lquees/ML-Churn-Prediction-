import sys


def configure_console():
    """
    Force stdout/stderr to UTF-8.

    Windows consoles default to cp1252, which cannot encode characters outside
    Latin-1. Any such character in a progress message kills an otherwise healthy
    run with UnicodeEncodeError on the first print. The pipeline's output is
    plain ASCII today, so this is defensive: it keeps a stray non-ASCII
    character in a message or a data value from taking a run down. Call it at
    the top of any entry point. No-op where the streams are already UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        # reconfigure() exists on TextIOWrapper (3.7+); pipes/captured streams
        # in some hosts are swapped for objects that lack it.
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
