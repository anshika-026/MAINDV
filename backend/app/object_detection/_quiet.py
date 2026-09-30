"""Routes the vendored modules' diagnostic print() calls to logging at DEBUG.

The detector printed ~10 status lines per frame to stdout, which on several
cameras floods the service log. Importing this `print` into each module
keeps every original message (visible with LOG_LEVEL=DEBUG) without editing
the ~220 call sites.
"""

import logging

_log = logging.getLogger("object_detection")


def print(*args, sep=" ", end="\n", file=None, flush=False):  # noqa: A001 - intentional shadow
    _log.debug(sep.join(str(a) for a in args))
