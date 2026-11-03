"""Request identity of an ``authoritative_catalog`` profile: the key and endpoint one discovery
request sends, captured once so its answer is cached under exactly that identity. Split out of
``hermes_cli.models``, whose fingerprint, discovery and disk-cache paths call into it.
"""

from __future__ import annotations

from typing import NamedTuple, Optional


class AccountRequest(NamedTuple):
    """The key and effective endpoint (configured relay, else the resolved base URL) of one request."""
    api_key: str
    endpoint: str


def account_request(normalized: str) -> Optional[AccountRequest]:
    """The identity a discovery for *normalized* sends now, or ``None`` when its profile does not
    declare ``authoritative_catalog`` (other providers keep their own fetchers and fingerprint)."""
    from hermes_cli import models
    from providers import get_provider_profile

    profile = get_provider_profile(normalized)
    if getattr(profile, "authoritative_catalog", False) is not True or profile.auth_type != "api_key":
        return None
    api_key, base_url = models._api_key_credentials(normalized)
    endpoint = models._configured_relay_base_url(normalized) or base_url or profile.base_url or ""
    return AccountRequest(api_key, endpoint.rstrip("/"))


def account_catalog(normalized: str, request: AccountRequest) -> Optional[list[str]]:
    """One discovery sent with *request*. Credentials are never re-resolved here: a key or endpoint
    rotated after the capture must not answer for the captured identity."""
    from hermes_cli import models
    from providers import get_provider_profile

    return models.probe_profile_catalog(
        normalized, get_provider_profile(normalized), request.api_key or None, request.endpoint or None)


def identity_parts(provider: str, request: Optional[AccountRequest]) -> list[str]:
    """Fingerprint parts the provider profile decides. External-process providers discover models
    through the launched program, so its command / argv env overrides identify the catalog. An
    authoritative catalog is an account grant: it belongs to the key and endpoint its request sends
    (*request*, or a fresh capture for a reader), which may come from $HERMES_HOME/.env, a
    request-scoped secret or the pool rather than the process environment."""
    import os

    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    if profile is not None and profile.auth_type == "external_process":
        evs = (*profile.process_command_env_vars, profile.process_args_env_var)
        return [f"{ev}={os.environ.get(ev, '')}" for ev in evs if ev]
    if (request := request or account_request(provider)) is None:
        return []
    return [f"effective_key={request.api_key}", f"effective_endpoint={request.endpoint}"]
