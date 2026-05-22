"""Quick connectivity check for LLM/Embedding endpoints.

This script validates:
1) Endpoint TCP reachability.
2) Optional lightweight API probes via LangChain clients.

Exit code:
- 0: all checks passed
- 1: any check failed
"""

from __future__ import annotations

import argparse
import socket
from typing import Any
from urllib.parse import urlparse

from modular_rag.core.settings import Settings, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check model endpoint connectivity.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument(
        "--network-timeout-sec",
        type=float,
        default=8.0,
        help="Timeout in seconds for TCP connectivity checks.",
    )
    parser.add_argument(
        "--skip-api-probe",
        action="store_true",
        help="Skip LLM/Embedding API probes and only run TCP checks.",
    )
    parser.add_argument(
        "--probe-target",
        choices=["all", "embedding", "llm"],
        default="all",
        help="Select which API probe to run (default: all).",
    )
    return parser.parse_args()


def _extract_endpoint_urls(settings: Settings) -> list[str]:
    candidates = [
        str(settings.llm.endpoint or settings.llm.api_url or "").strip(),
        str(settings.embedding.endpoint or settings.embedding.api_url or "").strip(),
    ]
    return [url for url in candidates if url]


def _preflight_tcp_connectivity(urls: list[str], timeout_sec: float) -> None:
    for raw_url in urls:
        parsed = urlparse(raw_url)
        host = parsed.hostname
        if not host:
            raise RuntimeError(f"Invalid endpoint URL (missing host): {raw_url}")
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80

        with socket.create_connection((host, port), timeout=timeout_sec):
            pass



def _build_langchain_models(settings: Settings) -> tuple[Any, Any]:
    provider = str(settings.llm.provider).strip().lower()
    emb_provider = str(settings.embedding.provider).strip().lower()

    if provider not in {"openai", "azure"} or emb_provider not in {"openai", "azure"}:
        raise ValueError("Only openai/azure providers are supported for API probing.")

    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
    except Exception as exc:
        raise ImportError("Missing dependency 'langchain-openai'.") from exc

    if provider == "azure":
        llm = AzureChatOpenAI(
            azure_deployment=settings.llm.deployment_name or settings.llm.model,
            api_version=settings.llm.api_version,
            azure_endpoint=settings.llm.endpoint or settings.llm.api_url,
            api_key=settings.llm.api_key,
            temperature=0,
            timeout=60.0,
            max_retries=2,
        )
    else:
        llm = ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_url,
            temperature=0,
            timeout=60.0,
            max_retries=2,
        )

    if emb_provider == "azure":
        embeddings = AzureOpenAIEmbeddings(
            model=settings.embedding.model,
            azure_deployment=settings.embedding.deployment_name or settings.embedding.model,
            api_version=settings.embedding.api_version,
            azure_endpoint=settings.embedding.endpoint or settings.embedding.api_url,
            api_key=settings.embedding.api_key,
        )
    else:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding.model,
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.api_url,
        )

    return llm, embeddings



def _format_probe_error(prefix: str, exc: Exception, *, endpoint: str, model: str) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    lowered = detail.lower()
    hint = ""

    if "404" in lowered or "notfound" in lowered or "page not found" in lowered:
        hint = (
            "Hint: endpoint or model is invalid for this API. "
            "Check api_url/base_url and model deployment mapping."
        )
    elif "401" in lowered or "unauthorized" in lowered or "invalid api key" in lowered:
        hint = "Hint: API key is invalid or missing permissions."
    elif "429" in lowered or "rate" in lowered:
        hint = "Hint: provider is rate-limiting; reduce traffic and retry later."
    elif "timeout" in lowered or "ssl" in lowered or "connection" in lowered:
        hint = "Hint: network/proxy/TLS instability; retry with larger timeout or check proxy." 

    msg = [
        f"[FAIL] {prefix} probe failed",
        f"  endpoint: {endpoint}",
        f"  model   : {model}",
        f"  error   : {detail}",
    ]
    if hint:
        msg.append(f"  {hint}")
    return "\n".join(msg)


def _probe_model_clients(llm: Any, embeddings: Any, *, settings: Settings, probe_target: str) -> None:
    emb_endpoint = str(settings.embedding.endpoint or settings.embedding.api_url or "")
    llm_endpoint = str(settings.llm.endpoint or settings.llm.api_url or "")

    if probe_target in {"all", "embedding"} and hasattr(embeddings, "embed_query"):
        try:
            embeddings.embed_query("connectivity probe")
            print("[OK] Embedding API probe passed")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                _format_probe_error(
                    "Embedding",
                    exc,
                    endpoint=emb_endpoint,
                    model=str(settings.embedding.model),
                )
            ) from exc

    if probe_target in {"all", "llm"} and hasattr(llm, "invoke"):
        try:
            llm.invoke("Reply with one word: ok")
            print("[OK] LLM API probe passed")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                _format_probe_error(
                    "LLM",
                    exc,
                    endpoint=llm_endpoint,
                    model=str(settings.llm.model),
                )
            ) from exc



def main() -> int:
    args = parse_args()
    try:
        if args.network_timeout_sec <= 0:
            raise ValueError("--network-timeout-sec must be > 0")

        settings = load_settings(args.settings)

        urls = _extract_endpoint_urls(settings)
        if not urls:
            raise RuntimeError("No endpoint URL found in settings (llm/api_url or embedding/api_url).")

        print(f"[INFO] TCP connectivity check: endpoints={len(urls)} timeout={args.network_timeout_sec}s")
        _preflight_tcp_connectivity(urls, timeout_sec=args.network_timeout_sec)
        print("[OK] TCP connectivity passed")

        if args.skip_api_probe:
            print("[INFO] API probe skipped by --skip-api-probe")
            return 0

        print("[INFO] API probe: building model clients...")
        llm, embeddings = _build_langchain_models(settings)
        print("[INFO] API probe: calling embedding and llm endpoints...")
        _probe_model_clients(llm=llm, embeddings=embeddings, settings=settings, probe_target=args.probe_target)
        print("[OK] API probe passed")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Connectivity check failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
