"""
manage.py — operator commands, run from backend/:

    python -m app.manage create-admin --email ops@example.com [--name "Ops"]
    python -m app.manage set-password --email ops@example.com
    python -m app.manage disable-admin --email ops@example.com
    python -m app.manage enable-admin --email ops@example.com
    python -m app.manage list-admins
    python -m app.manage purge-sessions
    python -m app.manage check-config
    python -m app.manage retention     # run the data-retention cleanup now

Passwords are read interactively (never from argv, where they'd land in
shell history and the process list) unless --password-stdin is given, which
reads one line from stdin for scripted provisioning.
"""

from __future__ import annotations

import argparse
import getpass
import sys


def _read_password(args) -> str:
    if args.password_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("New password: ")
    if first != getpass.getpass("Repeat password: "):
        raise SystemExit("Passwords do not match")
    return first


def main(argv: list[str] | None = None) -> int:
    from app import auth, config

    parser = argparse.ArgumentParser(prog="python -m app.manage")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("create-admin", "set-password"):
        p = sub.add_parser(name)
        p.add_argument("--email", required=True)
        p.add_argument("--password-stdin", action="store_true")
        if name == "create-admin":
            p.add_argument("--name")
    for name in ("disable-admin", "enable-admin"):
        sub.add_parser(name).add_argument("--email", required=True)
    sub.add_parser("list-admins")
    sub.add_parser("purge-sessions")
    sub.add_parser("check-config")
    sub.add_parser("retention")
    args = parser.parse_args(argv)

    if args.cmd == "check-config":
        problems = config.validate()
        print(f"APP_ENV={config.APP_ENV} DATA_DIR={config.DATA_DIR} MODEL_DIR={config.MODEL_DIR}")
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1 if problems else 0

    auth.init_db()
    try:
        if args.cmd == "create-admin":
            admin = auth.create_admin_user(args.email, _read_password(args), args.name)
            print(f"Created admin {admin['email']}")
        elif args.cmd == "set-password":
            auth.set_admin_password(args.email, _read_password(args))
            print("Password changed; that admin's existing sessions were signed out")
        elif args.cmd == "disable-admin":
            auth.set_admin_active(args.email, False)
            print("Disabled; that admin's sessions were signed out")
        elif args.cmd == "enable-admin":
            auth.set_admin_active(args.email, True)
            print("Enabled")
        elif args.cmd == "list-admins":
            for a in auth.list_admin_users():
                print(f"{a['email']:40} {'active' if a['is_active'] else 'DISABLED':9} {a['name']}")
        elif args.cmd == "retention":
            from app import retention

            for k, v in retention.run_once().items():
                print(f"{k:20} {v}")
        elif args.cmd == "purge-sessions":
            print(f"Removed {auth.purge_expired_sessions()} expired session(s)")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
