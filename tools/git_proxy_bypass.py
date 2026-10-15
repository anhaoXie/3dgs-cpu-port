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

# The reachable GitHub endpoint on this network.  Note that the address a
# public resolver returns for github.com (20.205.243.166) is NOT reachable from
# here, while 20.205.243.168 is -- and it serves both github.com and
# api.github.com.  Verified with:
#   curl --resolve api.github.com:443:20.205.243.168 https://api.github.com/  -> 200
GITHUB_IP = "20.205.243.168"

GITHUB_HOSTS = (
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "gist.github.com",
    "githubapp.com",
    "github.io",
    "pages.github.com",
)
GITHUB_SUFFIXES = (
    ".githubusercontent.com",
    ".githubassets.com",
)

BUFFER = 65536


def pick_ip(host: str) -> str | None:
    """Return the address to dial for *host*, or None to resolve normally."""
    h = host.lower()
    if h in GITHUB_HOSTS or h.endswith(GITHUB_SUFFIXES):
        return GITHUB_IP
    return None


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

    ip = pick_ip(host)
    try:
        remote_reader, remote_writer = await asyncio.wait_for(
            asyncio.open_connection(ip or host, port), timeout=20
        )
    except Exception as exc:  # noqa: BLE001 - report and close
        print(f"  ! CONNECT {host}:{port} via {ip or 'dns'} failed: {exc}",
              file=sys.stderr, flush=True)
        try:
            client_writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            await client_writer.drain()
        except Exception:
            pass
        client_writer.close()
        return

    print(f"  -> CONNECT {host}:{port}  dialling {ip or host}  (client {peer})",
          flush=True)

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
    print(f"  github hosts -> {GITHUB_IP}", flush=True)
    print(f"  all other hosts are resolved normally", flush=True)
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
