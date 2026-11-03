"""``ProviderProfile.authoritative_catalog``: a successful account list, empty included, is the
catalog every surface offers, while a curated provider keeps merging its ``fallback_models``.

Both providers are a real plugin discovered from a temp HERMES_HOME and answer from a local
``/models`` server, so credentials, fingerprints and the disk cache run unpatched. The endpoint and
late-response tests also cover the curated provider: those identity bugs are not flag-specific.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli import models
from hermes_cli.models_catalog_static import AuthoritativeModels, CuratedFallbackModels
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tests.waiting import wait_until

ACCOUNT, CURATED = "zz-account", "zz-curated"
SEED = ["seed-a", "seed-b"]
KEY = "account-key-1"
REPO = Path(__file__).resolve().parents[2]

# The curated profile registers first, so on a core without the field only the flagged one is lost.
_PLUGIN = '''
from providers import register_provider
from providers.base import ProviderProfile

for name, flag in (("zz-curated", {}), ("zz-account", {"authoritative_catalog": True})):
    env = name.upper().replace("-", "_")
    canonical = f"https://{name}.invalid/v1"
    register_provider(ProviderProfile(
        name=name, env_vars=(f"{env}_KEY", f"{env}_BASE_URL"), base_url=canonical,
        fallback_models=("seed-a", "seed-b"),
        provider_overlay={"base_url_override": canonical, "extra_env_vars": (f"{env}_KEY",),
                          "base_url_env_var": f"{env}_BASE_URL"},
        **flag))
'''


class Gateway:
    """``/models`` answering per bearer key: a model list or an HTTP status. While ``release`` is
    set, each request records its key in ``waiting`` and blocks until the event fires."""

    def __init__(self):
        self.accounts: dict = {}
        self.hits: list[str] = []
        self.waiting: list[str] = []
        self.release: threading.Event | None = None
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                key = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
                answer = gateway.accounts.get(key, 401)
                gateway.hits.append(key)
                if gateway.release is not None:
                    gateway.waiting.append(key)
                    gateway.release.wait(30)
                if isinstance(answer, int):
                    self.send_response(answer)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.dumps({"object": "list", "data": [{"id": m, "object": "model"} for m in answer]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        if self.release is not None:
            self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(timeout=5)


@pytest.fixture
def gateway():
    server = Gateway()
    yield server
    server.close()


@pytest.fixture
def relay():
    server = Gateway()
    yield server
    server.close()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    plugin = home / "plugins" / "model-providers" / "zz-catalog"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: zz-catalog\nkind: model-provider\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(_PLUGIN, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    with models._swr_refresh_lock:
        models._swr_refresh_inflight.clear()
    yield home
    reset_hermes_home_override(token)


def env_names(slug):
    prefix = slug.upper().replace("-", "_")
    return f"{prefix}_KEY", f"{prefix}_BASE_URL"


def use(monkeypatch, slug, gateway, key):
    key_env, base_env = env_names(slug)
    monkeypatch.setenv(key_env, key)
    monkeypatch.setenv(base_env, gateway.base_url)


def refresh(slug):
    return models.cached_provider_model_ids(slug, force_refresh=True)


def picker(slug, **kwargs):
    """The ``/model`` and GUI picker read, with the curated row ``_build_curated_lists`` gives a
    plugin profile (its ``fallback_models``)."""
    from hermes_cli.model_switch_providers import _live_or_curated_ids

    return _live_or_curated_ids(slug, {slug: list(SEED)}, merge_models_dev=False, **kwargs)


def picker_after_restart(home, slug):
    script = (
        "import json, sys\n"
        "from hermes_cli.model_switch_providers import _live_or_curated_ids\n"
        "slug = sys.argv[1]\n"
        "print(json.dumps(_live_or_curated_ids(slug, {slug: ['seed-a', 'seed-b']}, merge_models_dev=False)))\n"
    )
    env = {name: os.environ[name] for name in ("PATH", "HOME", *env_names(slug))}
    result = subprocess.run(
        [sys.executable, "-c", script, slug], cwd=REPO, env={**env, "HERMES_HOME": str(home)},
        capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def configure_relay(slug, endpoint):
    from hermes_cli.config import save_config

    model = {"provider": slug, "default": "seed-a"}
    save_config({"model": {**model, "base_url": endpoint.base_url} if endpoint else model})


def test_a_curated_provider_still_merges_its_fallback_models(monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, CURATED, gateway, KEY)
    assert models.provider_model_ids(CURATED, force_refresh=True) == [*SEED, "live-x"]

    gateway.accounts[KEY] = []
    empty = models.provider_model_ids(CURATED, force_refresh=True)
    assert isinstance(empty, CuratedFallbackModels) and empty == SEED


def test_an_authoritative_account_list_replaces_the_fallback_models(monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, ACCOUNT, gateway, KEY)

    live = models.provider_model_ids(ACCOUNT, force_refresh=True)
    assert isinstance(live, AuthoritativeModels) and live == ["live-x"]
    assert refresh(ACCOUNT) == ["live-x"]
    assert picker(ACCOUNT) == ["live-x"]


def test_an_empty_account_list_replaces_a_warm_catalog_and_survives_a_restart(home, monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, ACCOUNT, gateway, KEY)
    assert refresh(ACCOUNT) == ["live-x"]

    gateway.accounts[KEY] = []
    assert refresh(ACCOUNT) == []
    hits = len(gateway.hits)
    assert picker(ACCOUNT) == []
    assert picker(ACCOUNT, non_blocking=True) == []
    assert picker_after_restart(home, ACCOUNT) == []
    assert len(gateway.hits) == hits


def test_a_cold_home_caches_an_empty_account_list(home, monkeypatch, gateway):
    gateway.accounts[KEY] = []
    use(monkeypatch, ACCOUNT, gateway, KEY)

    assert picker(ACCOUNT) == []
    assert len(gateway.hits) == 1
    assert picker_after_restart(home, ACCOUNT) == []
    assert len(gateway.hits) == 1


def test_a_failed_fetch_keeps_the_same_identity_catalog_with_one_request(monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, ACCOUNT, gateway, KEY)
    assert refresh(ACCOUNT) == ["live-x"]

    gateway.accounts[KEY] = 503
    hits = len(gateway.hits)
    assert refresh(ACCOUNT) == ["live-x"]
    assert len(gateway.hits) == hits + 1
    assert picker(ACCOUNT) == ["live-x"]

    gateway.accounts[KEY] = []
    assert refresh(ACCOUNT) == []
    gateway.accounts[KEY] = 503
    assert refresh(ACCOUNT) == []
    assert picker(ACCOUNT) == []


def test_a_failed_fetch_without_a_catalog_offers_the_fallback_models(monkeypatch, gateway):
    gateway.accounts[KEY] = 503
    use(monkeypatch, ACCOUNT, gateway, KEY)

    assert refresh(ACCOUNT) == SEED
    assert picker(ACCOUNT) == SEED


def test_an_empty_relay_list_is_the_answer(monkeypatch, gateway, relay):
    gateway.accounts[KEY] = ["canonical-x"]
    relay.accounts[KEY] = []
    use(monkeypatch, ACCOUNT, gateway, KEY)
    configure_relay(ACCOUNT, relay)

    assert refresh(ACCOUNT) == []
    assert picker(ACCOUNT) == []
    assert gateway.hits == []


@pytest.mark.parametrize("slug", [CURATED, ACCOUNT])
def test_a_configured_endpoint_switch_never_serves_the_previous_endpoints_rows(monkeypatch, gateway, relay, slug):
    gateway.accounts[KEY] = 503
    relay.accounts[KEY] = ["relay-private"]
    use(monkeypatch, slug, gateway, KEY)
    configure_relay(slug, relay)
    assert "relay-private" in refresh(slug)

    configure_relay(slug, None)
    assert "relay-private" not in models.cached_provider_model_ids(slug)
    assert "relay-private" not in picker(slug)

    configure_relay(slug, relay)
    assert "relay-private" in picker(slug)


@pytest.mark.parametrize("slug", [CURATED, ACCOUNT])
def test_a_refresh_answer_is_filed_under_the_identity_it_was_fetched_for(monkeypatch, gateway, slug):
    gateway.accounts["key-a"] = ["a-private"]
    use(monkeypatch, slug, gateway, "key-a")
    stale = time.time() - 2 * models._PROVIDER_MODELS_CACHE_TTL
    models._store_cache_entry(slug, models._cache_entry(models._credential_fingerprint(slug), ["a-old"], stale))
    gateway.release = threading.Event()

    assert models.cached_provider_model_ids(slug) == ["a-old"]
    wait_until(lambda: gateway.waiting == ["key-a"], what="the revalidation request under key-a")
    use(monkeypatch, slug, gateway, "key-b")
    gateway.release.set()
    wait_until(lambda: not models._swr_refresh_inflight, what="the revalidation to land")
    gateway.release = None

    assert "a-private" not in models.cached_provider_model_ids(slug)
    assert "a-private" not in picker(slug)


def test_first_setup_offers_the_account_list_before_models_dev(monkeypatch, gateway):
    from hermes_cli import model_setup_flows as flows
    from hermes_cli.auth import PROVIDER_REGISTRY

    # models.dev stand-in: the flagged provider must not consult it, the curated one still does.
    monkeypatch.setattr(flows, "_models_dev_merged", lambda provider_id, curated: ["models-dev-row"])

    def setup_list(slug):
        return flows._api_key_provider_model_list(slug, PROVIDER_REGISTRY[slug], KEY, env_names(slug)[0], gateway.base_url)

    use(monkeypatch, ACCOUNT, gateway, KEY)
    gateway.accounts[KEY] = []
    assert setup_list(ACCOUNT) == []
    gateway.accounts[KEY] = ["live-x"]
    assert setup_list(ACCOUNT) == ["live-x"]
    gateway.accounts[KEY] = 503
    hits = len(gateway.hits)
    assert setup_list(ACCOUNT) == SEED
    assert len(gateway.hits) == hits + 1

    use(monkeypatch, CURATED, gateway, KEY)
    assert setup_list(CURATED) == ["models-dev-row"]


@pytest.mark.parametrize("warm", [["live-x"], []], ids=["private", "empty"])
def test_a_failed_setup_probe_keeps_the_same_identity_list(monkeypatch, gateway, warm):
    from hermes_cli import model_setup_flows as flows
    from hermes_cli.auth import PROVIDER_REGISTRY

    def setup_list(key):
        return flows._api_key_provider_model_list(
            ACCOUNT, PROVIDER_REGISTRY[ACCOUNT], key, env_names(ACCOUNT)[0], gateway.base_url)

    gateway.accounts.update({KEY: warm, "account-key-2": 503})
    use(monkeypatch, ACCOUNT, gateway, KEY)
    assert refresh(ACCOUNT) == warm
    gateway.accounts[KEY] = 503
    assert setup_list(KEY) == warm
    assert setup_list("account-key-2") == SEED


@pytest.fixture
def offline_registries(monkeypatch):
    """Only the remote registries the public picker also reads are stubbed; every row builder,
    credential resolver and the disk cache run as shipped."""
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda **kwargs: {})
    monkeypatch.setattr(models, "get_curated_nous_model_ids", list)
    monkeypatch.setattr(models, "fetch_ollama_cloud_models", lambda **kwargs: [])


def public_rows(**kwargs):
    """``list_authenticated_providers``, the ``/model`` and GUI picker's public entry point."""
    from hermes_cli.model_switch_providers import list_authenticated_providers
    from providers import list_providers

    excluded = [p.name for p in list_providers() if p.name not in (ACCOUNT, CURATED)]
    rows = list_authenticated_providers(
        current_provider=ACCOUNT, excluded_providers=excluded, probe_custom_providers=False,
        for_picker=True, **kwargs)
    return {row["slug"]: row["models"] for row in rows}


def test_a_public_refresh_shows_removed_models_and_an_empty_account_list(monkeypatch, gateway, offline_registries):
    gateway.accounts[KEY] = ["live-x", "live-y"]
    use(monkeypatch, ACCOUNT, gateway, KEY)
    use(monkeypatch, CURATED, gateway, KEY)
    assert public_rows(refresh=True)[ACCOUNT] == ["live-x", "live-y"]

    gateway.accounts[KEY] = ["live-x"]
    rows = public_rows(refresh=True)
    assert rows[ACCOUNT] == ["live-x"]
    assert set(SEED) <= set(rows[CURATED])

    gateway.accounts[KEY] = []
    rows = public_rows(refresh=True)
    assert rows[ACCOUNT] == []
    assert set(SEED) <= set(rows[CURATED])


@pytest.mark.parametrize("warm", [["live-x"], []], ids=["private", "empty"])
def test_a_failed_public_refresh_keeps_the_same_identity_list(monkeypatch, gateway, offline_registries, warm):
    gateway.accounts.update({KEY: warm, "curated-key": ["live-c"]})
    use(monkeypatch, ACCOUNT, gateway, KEY)
    use(monkeypatch, CURATED, gateway, "curated-key")
    rows = public_rows(refresh=True)
    assert rows[ACCOUNT] == warm
    assert "live-c" in rows[CURATED]

    gateway.accounts.update({KEY: 503, "curated-key": 503})
    rows = public_rows(refresh=True)
    assert rows[ACCOUNT] == warm
    assert gateway.hits.count(KEY) == 2
    # A curated provider's failed refresh still offers its curated list, as before.
    assert rows[CURATED] == SEED


@pytest.mark.parametrize("source", ["environ", "dotenv", "scope"])
def test_a_rotated_effective_key_never_reuses_the_account_list(home, monkeypatch, gateway, offline_registries, source):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    key_env = env_names(ACCOUNT)[0]

    def rotate(key):
        if source == "environ":
            monkeypatch.setenv(key_env, key)
        elif source == "dotenv":
            (home / ".env").write_text(f"{key_env}={key}\n", encoding="utf-8")
        else:
            return set_secret_scope({key_env: key})
        return None

    gateway.accounts.update({KEY: ["a-private"], "account-key-2": 503, "curated-key": ["c-live"]})
    use(monkeypatch, ACCOUNT, gateway, "process-key-0")
    use(monkeypatch, CURATED, gateway, "curated-key")
    tokens = [rotate(KEY)]
    try:
        assert public_rows(refresh=True)[ACCOUNT] == ["a-private"]
        tokens.append(rotate("account-key-2"))
        rows = public_rows()
        assert rows[ACCOUNT] == SEED
        assert "c-live" in rows[CURATED] and "a-private" not in rows[CURATED]
        assert public_rows(refresh=True)[ACCOUNT] == SEED
        assert gateway.hits[-1] == "account-key-2"
    finally:
        for token in reversed(tokens):
            if token is not None:
                reset_secret_scope(token)


def test_a_rotated_endpoint_never_reuses_the_account_list(monkeypatch, gateway, relay, offline_registries):
    gateway.accounts[KEY] = ["a-private"]
    relay.accounts[KEY] = 503
    use(monkeypatch, ACCOUNT, gateway, KEY)
    assert public_rows(refresh=True)[ACCOUNT] == ["a-private"]

    use(monkeypatch, ACCOUNT, relay, KEY)
    assert public_rows()[ACCOUNT] == SEED
    use(monkeypatch, ACCOUNT, gateway, KEY)
    configure_relay(ACCOUNT, relay)
    assert public_rows()[ACCOUNT] == SEED
    assert relay.hits == [KEY, KEY]


def test_a_late_answer_for_a_previous_scoped_key_is_not_shown(monkeypatch, gateway, offline_registries):
    from agent.secret_scope import reset_secret_scope, set_secret_scope

    key_env = env_names(ACCOUNT)[0]
    gateway.accounts.update({KEY: ["a-private"], "account-key-2": 503})
    use(monkeypatch, ACCOUNT, gateway, "process-key-0")
    tokens = [set_secret_scope({key_env: KEY})]
    try:
        stale = time.time() - 2 * models._PROVIDER_MODELS_CACHE_TTL
        models._store_cache_entry(ACCOUNT, models._cache_entry(models._credential_fingerprint(ACCOUNT), ["a-old"], stale))
        gateway.release = threading.Event()
        assert public_rows()[ACCOUNT] == ["a-old"]
        wait_until(lambda: gateway.waiting == [KEY], what="the revalidation request under the first scoped key")
        tokens.append(set_secret_scope({key_env: "account-key-2"}))
        gateway.release.set()
        wait_until(lambda: not models._swr_refresh_inflight, what="the revalidation to land")
        gateway.release = None

        assert "a-private" not in public_rows()[ACCOUNT]
    finally:
        for token in reversed(tokens):
            reset_secret_scope(token)


def test_a_key_rotated_after_the_identity_capture_never_files_its_grant_under_the_old_key(monkeypatch, gateway):
    """A to B right after the fingerprint is taken, then back to A: the request still goes out under
    A, so B's grant can never be cached under A's identity."""
    key_env = env_names(ACCOUNT)[0]
    gateway.accounts.update({"account-a": ["grant-a"], "account-b": ["grant-b"]})
    use(monkeypatch, ACCOUNT, gateway, "account-a")
    assert refresh(ACCOUNT) == ["grant-a"]
    gateway.hits.clear()
    fingerprint = models._credential_fingerprint

    def rotate_after_capture(provider, *args):
        captured = fingerprint(provider, *args)
        monkeypatch.setenv(key_env, "account-b")
        return captured

    monkeypatch.setattr(models, "_credential_fingerprint", rotate_after_capture)
    during = refresh(ACCOUNT)
    monkeypatch.setattr(models, "_credential_fingerprint", fingerprint)
    assert gateway.hits == ["account-a"] and during == ["grant-a"]

    assert models.cached_provider_model_ids(ACCOUNT) == ["grant-b"]
    monkeypatch.setenv(key_env, "account-a")
    assert models.cached_provider_model_ids(ACCOUNT) == ["grant-a"]
    assert picker(ACCOUNT) == ["grant-a"]


@pytest.mark.parametrize("rotated", ["key", "endpoint"])
def test_an_identity_rotated_while_the_request_is_in_flight_keeps_each_grant_with_its_identity(
        monkeypatch, gateway, relay, offline_registries, rotated):
    import contextvars

    key_env, base_env = env_names(ACCOUNT)
    gateway.accounts.update({"account-a": ["grant-a"], "account-b": ["grant-b"]})
    relay.accounts.update({"account-a": ["relay-grant"]})
    use(monkeypatch, ACCOUNT, gateway, "account-a")
    gateway.release = threading.Event()
    ctx = contextvars.copy_context()
    worker = threading.Thread(target=lambda: ctx.run(refresh, ACCOUNT), daemon=True)
    worker.start()
    wait_until(lambda: gateway.waiting == ["account-a"], what="the request under the first identity")
    if rotated == "key":
        monkeypatch.setenv(key_env, "account-b")
    else:
        monkeypatch.setenv(base_env, relay.base_url)
    gateway.release.set()
    worker.join(timeout=30)
    gateway.release = None

    second = ["grant-b"] if rotated == "key" else ["relay-grant"]
    assert public_rows()[ACCOUNT] == second
    monkeypatch.setenv(key_env, "account-a")
    monkeypatch.setenv(base_env, gateway.base_url)
    assert public_rows()[ACCOUNT] == ["grant-a"]


def test_a_failed_refresh_of_a_mixed_cache_keeps_only_the_account_list(monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, ACCOUNT, gateway, KEY)
    use(monkeypatch, CURATED, gateway, KEY)
    assert refresh(ACCOUNT) == ["live-x"]
    assert refresh(CURATED) == [*SEED, "live-x"]

    gateway.accounts[KEY] = 503
    models.expire_provider_models_cache()
    assert models.cached_provider_model_ids(ACCOUNT) == ["live-x"]
    assert models.cached_provider_model_ids(CURATED) == SEED


def test_removing_a_credential_still_drops_its_account_list(monkeypatch, gateway):
    gateway.accounts[KEY] = ["live-x"]
    use(monkeypatch, ACCOUNT, gateway, KEY)
    assert refresh(ACCOUNT) == ["live-x"]

    models.clear_provider_models_cache(ACCOUNT)
    gateway.accounts[KEY] = 503
    assert models.cached_provider_model_ids(ACCOUNT) == SEED
