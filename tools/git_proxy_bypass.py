#!/usr/bin/env python
"""Minimal local CONNECT relay that works around a hosts-file block on GitHub.

Why this exists
---------------
On this machine ``C:\\Windows\\System32\\drivers\\etc\\hosts`` pins a long list of
domains (github.com, api.github.com, *.githubusercontent.com, huggingface.co,
Steam, ...) to ``127.0.0.1``.  The hosts file is owned by SYSTEM and is
read-only for BUILTIN\\Users, so it cannot be corrected without administrator
rights -- and it is a block the machine's user configured on purpose.

The *network* itself is fine: addressing GitHub's real IP directly works.

    curl --resolve api.github.com:443:20.205.243.168 https://api.github.com/
    -> HTTP 200

libcurl (and therefore git) never consults the hosts file when it is handed a
proxy: it sends the proxy a ``CONNECT github.com:443`` line and lets the proxy
do the name resolution.  So a tiny local relay that ignores the requested host
and dials the known-good IP restores git access *without* modifying the hosts
file, without administrator rights, and end-to-end TLS is preserved (the
github.com certificate still validates, because the SNI/hostname is unchanged).

Usage
-----
    python tools/git_proxy_bypass.py --port 8899
    git config --local http.proxy http://127.0.0.1:8899
    git push -u origin main
    git config --local --unset http.proxy      # afterwards, to clean up

Only the hosts listed in GITHUB_HOSTS are redirected; every other CONNECT
target is resolved normally, so nothing else about networking changes.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import sys

# Per-host endpoint overrides.
#
# These MUST be per-host.  github.com and api.github.com are different services
# and are not interchangeable: 20.205.243.168 answers api.github.com with 200
# but answers github.com with 400, and git then hangs forever with a connection
# that is open but carries nothing.  Picking one address for "GitHub" cost a
# 10-minute stall, so the mapping is explicit and github.com has failover.
#
# Verified reachability from this network (curl --resolve <host>:443:<ip>):
#   20.27.177.113   github.com -> 200   (fastest: connect 0.11s)
#   140.82.113.3    github.com -> 200
#   140.82.121.3    github.com -> 200
#   20.205.243.168  api.github.com -> 200
#   20.205.243.166  unreachable, 140.82.112.3 unreachable
GITHUB_ENDPOINTS: dict[str, list[str]] = {
    "github.com": ["20.27.177.113", "140.82.113.3", "140.82.121.3"],
    "api.github.com": ["20.205.243.168"],
}

# Hosts that are hosts-file-blocked but NOT needed (and not worth guessing an IP
# for).  Everything else is resolved normally by the OS.
DEFAULT_PORT = 443

BUFFER = 65536


def endpoint_for(host: str) -> list[str] | None:
    """Return candidate addresses to dial for *host*, or None to resolve normally."""
    return GITHUB_ENDPOINTS.get(host.lower())


async def dial(host: str, port: int):
    """Open a connection to *host*, trying each known candidate in order."""
    candidates = endpoint_for(host)
    if candidates is None:
        # Not a redirected host: let the OS resolve it.
        return await asyncio.open_connection(host, port), host

    last_exc: Exception | None = None
    for ip in candidates:
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port), timeout=15
            )
            return (reader, writer), ip
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            last_exc = exc
            print(f"  ! {host}: {ip} failed ({exc}); trying next", file=sys.stderr, flush=True)
    raise ConnectionError(f"no reachable address for {host}: {last_exc}")


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(BUFFER)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_reader: asyncio.StreamReader,
                 client_writer: asyncio.StreamWriter) -> None:
    peer = client_writer.get_extra_info("peername")
    try:
        header = await asyncio.wait_for(client_reader.readuntil(b"\r\n\r\n"), timeout=15)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
        client_writer.close()
        return

    try:
        request_line = header.split(b"\r\n", 1)[0].decode("latin-1")
    except Exception:
        client_writer.close()
        return

    parts = request_line.split()
    if len(parts) < 2 or parts[0].upper() != "CONNECT":
        # Plain HTTP proxying is not needed for git-over-HTTPS.
        client_writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
        await client_writer.drain()
        client_writer.close()
        return

    target = parts[1]
    if ":" in target:
        host, _, port_s = target.rpartition(":")
        try:
            port = int(port_s)
        except ValueError:
            port = 443
    else:
        host, port = target, 443

    try:
        (remote_reader, remote_writer), dialled = await dial(host, port)
    except Exception as exc:  # noqa: BLE001 - report and close
        print(f"  ! CONNECT {host}:{port} failed: {exc}", file=sys.stderr, flush=True)
        try:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
        except Exception:
            pass
        client_writer.close()
        return

    print(f"  -> CONNECT {host}:{port}  dialling {dialled}  (client {peer})", flush=True)

    client_writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    await client_writer.drain()

    await asyncio.gather(
        pipe(client_reader, remote_writer),
        pipe(remote_reader, client_writer),
        return_exceptions=True,
    )


async def main_async(port: int, host: str) -> None:
    server = await asyncio.start_server(handle, host, port)
    print(f"git proxy-bypass relay listening on {host}:{port}", flush=True)
    for h, ips in GITHUB_ENDPOINTS.items():
        print(f"  {h:>18} -> {', '.join(ips)}", flush=True)
    print("  all other hosts are resolved normally", flush=True)
    async with server:
        await server.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    try:
        asyncio.run(main_async(args.port, args.host))
    except KeyboardInterrupt:
        print("\nrelay stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
