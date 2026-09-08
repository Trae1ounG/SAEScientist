#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from urllib.request import ProxyHandler, Request, build_opener


PROBE_URL = os.environ["SAE_PROBE_URL"].rstrip("/") + "/probe"
OPENER = build_opener(ProxyHandler({}))


def write(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def call_probe(arguments: dict) -> dict:
    body = json.dumps(arguments).encode("utf-8")
    request = Request(
        PROBE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with OPENER.open(request, timeout=180) as response:
        return json.loads(response.read())


TOOL = {
    "name": "probe_sae",
    "description": (
        "Run candidate texts through the hidden SAE. Returns only measured feature IDs, "
        "activation values, and ranks; it exposes no public labels or expert metadata."
    ),
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    "inputSchema": {
        "type": "object",
        "properties": {
            "texts": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 4096},
                "minItems": 1,
                "maxItems": 64,
            },
            "top_k": {"type": "integer", "minimum": 1, "maximum": 256},
            "feature_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 131071},
                "maxItems": 256,
            },
        },
        "required": ["texts"],
        "additionalProperties": False,
    },
}


for line in sys.stdin:
    try:
        message = json.loads(line)
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": message.get("params", {}).get(
                            "protocolVersion", "2024-11-05"
                        ),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "sae-probe", "version": "0.1.0"},
                    },
                }
            )
        elif method == "tools/list":
            write({"jsonrpc": "2.0", "id": request_id, "result": {"tools": [TOOL]}})
        elif method == "tools/call":
            params = message.get("params", {})
            if params.get("name") != TOOL["name"]:
                raise ValueError("unknown tool")
            result = call_probe(params.get("arguments", {}))
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, ensure_ascii=False),
                            }
                        ],
                        "structuredContent": result,
                    },
                }
            )
        elif method == "ping":
            write({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif request_id is not None:
            write({"jsonrpc": "2.0", "id": request_id, "result": {}})
    except Exception as exc:
        if "request_id" in locals() and request_id is not None:
            write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32000, "message": str(exc)},
                }
            )
        else:
            print(str(exc), file=sys.stderr, flush=True)

