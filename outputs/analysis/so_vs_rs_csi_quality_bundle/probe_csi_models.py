#!/usr/bin/env python3
"""Probe CSI model metadata and model-specific thinking controls."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = ("Qwen3.8-Max", "DeepSeek-V4-Pro", "GLM-5.3")
DEFAULT_OPTIONS = ROOT / "model_request_options.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default="http://113.46.219.251:8080/v1"
    )
    parser.add_argument("--api-key-env", default="CSI_API_KEY")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reasoning-matrix", action="store_true")
    parser.add_argument("--model-options-file", type=Path, default=DEFAULT_OPTIONS)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "csi_model_probe.json"
    )
    return parser.parse_args()


def request_json(
    url: str,
    api_key: str,
    timeout: float,
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "ok": True,
                "http_status": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "http_status": exc.code,
            "error": exc.read().decode("utf-8", errors="replace")[:4000],
        }
    except Exception as exc:
        return {"ok": False, "http_status": None, "error": str(exc)}


def iso_timestamp(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def summarize_response(
    model: str,
    case: str,
    options: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "requested_model": model,
        "case": case,
        "request_options": options,
        "ok": result["ok"],
        "http_status": result["http_status"],
    }
    if not result["ok"]:
        summary["error"] = result.get("error")
        return summary
    body = result["body"]
    choices = body.get("choices") if isinstance(body, dict) else None
    message = (
        choices[0].get("message", {})
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    summary.update(
        {
            "returned_model": body.get("model"),
            "response_created": body.get("created"),
            "response_created_utc": iso_timestamp(body.get("created")),
            "system_fingerprint": body.get("system_fingerprint"),
            "response_keys": sorted(body),
            "message_keys": sorted(message) if isinstance(message, dict) else [],
            "has_reasoning_content": bool(
                isinstance(message, dict) and message.get("reasoning_content")
            ),
            "usage": body.get("usage"),
        }
    )
    return summary


def probe_cases(
    model: str,
    configured: dict[str, Any],
    matrix: bool,
) -> list[tuple[str, dict[str, Any]]]:
    if not matrix:
        return [("configured", configured)]
    if model == "Qwen3.8-Max":
        return [
            ("provider_default", {}),
            ("thinking_disabled", {"enable_thinking": False}),
            (
                "thinking_budget_8192",
                {"enable_thinking": True, "thinking_budget": 8192},
            ),
        ]
    if model in {"DeepSeek-V4-Pro", "GLM-5.3"}:
        return [("provider_default", {})] + [
            (
                f"thinking_{effort}",
                {
                    "thinking": {"type": "enabled"},
                    "reasoning_effort": effort,
                },
            )
            for effort in ("low", "high", "max")
        ]
    return [("provider_default", {}), ("configured", configured)]


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    base_url = args.base_url.rstrip("/")
    raw_options = json.loads(args.model_options_file.read_text(encoding="utf-8"))
    if not isinstance(raw_options, dict):
        raise ValueError("model options file must contain a JSON object")
    catalog = request_json(
        base_url + "/models", api_key, args.timeout
    )
    probes: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for model in args.models:
        configured = raw_options.get(model, {})
        if not isinstance(configured, dict):
            raise ValueError(f"invalid options for {model}")
        for case, options in probe_cases(model, configured, args.reasoning_matrix):
            payload: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Return exactly one JSON object: "
                            '{"probe":"ok"}'
                        ),
                    }
                ],
                "temperature": 0,
                "max_tokens": 64,
                "response_format": {"type": "json_object"},
            }
            payload.update(options)
            result = request_json(
                base_url + "/chat/completions",
                api_key,
                args.timeout,
                payload=payload,
            )
            probes.append(summarize_response(model, case, options, result))
            raw.append(
                {
                    "requested_model": model,
                    "case": case,
                    "request_options": options,
                    "result": result,
                }
            )
            print(
                f"{model} case={case}: "
                f"status={result.get('http_status')} ok={result.get('ok')}",
                flush=True,
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": base_url,
        "models": list(args.models),
        "configured_model_options": raw_options,
        "reasoning_matrix": args.reasoning_matrix,
        "catalog": catalog,
        "probes": probes,
        "interpretation": {
            "response_created": "request/response creation time, not model release date",
            "catalog_created": (
                "provider metadata only; treat as a release date only if the provider "
                "explicitly documents that meaning"
            ),
            "exact_backend_version": (
                "known only when returned_model, system_fingerprint, or provider "
                "catalog exposes an immutable revision"
            ),
        },
        "raw_results": raw,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
