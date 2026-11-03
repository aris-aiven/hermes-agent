"""Provider discovery must enforce the same version gate before either import route."""

from pathlib import Path
import sys

import pytest

import providers
from hermes_cli import plugins_manifest
from hermes_cli.plugin_host_profiles import is_hosted_profile


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(providers, "_HOME_LAYERS", {})
    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "0.22.6")
    modules_before = set(sys.modules)
    yield tmp_path
    for name in set(sys.modules) - modules_before:
        if name.startswith(
            ("_hermes_user_provider_", "plugins.model_providers.compat_")
        ):
            sys.modules.pop(name, None)


def write_plugin(directory, manifest):
    directory.mkdir(parents=True)
    marker = directory / "imported"
    (directory / "__init__.py").write_text(
        "from pathlib import Path\n"
        "from providers import register_provider\n"
        "from providers.base import ProviderProfile\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "register_provider(ProviderProfile(name='compat-probe', "
        "aliases=('compat-alias',), base_url='https://compat.test/v1'))\n",
        encoding="utf-8",
    )
    if manifest:
        filename, content = manifest
        (directory / filename).write_text(content, encoding="utf-8")
    return marker


@pytest.mark.parametrize("layout", ["flat", "nested"])
@pytest.mark.parametrize("filename", ["plugin.yaml", "plugin.yml"])
@pytest.mark.parametrize("mode", ["in_process", "host"])
@pytest.mark.parametrize(
    ("spec", "allowed"),
    [
        ("", True),
        (">=0.22.6", True),
        (">0.22.6", False),
        ("<=0.22.6", True),
        ("<0.22.6", False),
        ("==0.22.6", True),
        ("!=0.22.6", False),
        (">=0.22.0,<0.23.0", True),
        (">=0.22.6,<0.22.6", False),
        ("0.22.7", False),
    ],
)
def test_native_discovery_version_equivalence(
    home, layout, filename, mode, spec, allowed
):
    (home / "config.yaml").write_text(
        f"plugins:\n  isolation: {mode}\n", encoding="utf-8"
    )
    root = home / "plugins"
    directory = (
        root / "compat-probe"
        if layout == "flat"
        else root / "model-providers/compat-probe"
    )
    declaration = "name: compat-probe\nkind: model-provider\n"
    if spec:
        declaration += f"requires_hermes: '{spec}'\n"
    marker = write_plugin(directory, (filename, declaration))

    profile = providers.get_provider_profile("compat-probe")
    assert (profile is not None) is allowed
    assert marker.exists() is allowed
    assert (providers.get_provider_profile("compat-alias") is not None) is allowed
    if allowed:
        assert is_hosted_profile(profile) is (mode == "host")
        assert providers.plugin_load_error("compat-probe") == ""
    else:
        error = providers.plugin_load_error("compat-probe")
        assert "requires hermes" in error and spec in error and "0.22.6" in error
    assert providers.get_provider_profile("openrouter") is not None


@pytest.mark.parametrize("mode", ["in_process", "host"])
def test_manifestless_nested_provider_keeps_legacy_admission(home, mode):
    (home / "config.yaml").write_text(
        f"plugins:\n  isolation: {mode}\n", encoding="utf-8"
    )
    marker = write_plugin(home / "plugins/model-providers/compat-probe", None)
    assert providers.get_provider_profile("compat-probe") is not None
    assert marker.exists()


@pytest.mark.parametrize("filename", ["plugin.yaml", "plugin.yml"])
@pytest.mark.parametrize("declaration", ["kind: [not valid\n", "- not-a-mapping\n"])
def test_invalid_nested_manifest_refuses_before_import(home, filename, declaration):
    marker = write_plugin(
        home / "plugins/model-providers/compat-probe", (filename, declaration)
    )
    assert providers.get_provider_profile("compat-probe") is None
    assert not marker.exists()
    assert "invalid" in providers.plugin_load_error("compat-probe")


@pytest.mark.parametrize(
    ("primary", "secondary", "allowed"),
    [
        (">0.22.6", "==0.22.6", False),
        ("==0.22.6", ">0.22.6", True),
    ],
)
def test_yaml_takes_precedence_when_both_native_manifest_names_exist(
    home, primary, secondary, allowed
):
    directory = home / "plugins/model-providers/compat-probe"
    marker = write_plugin(
        directory,
        ("plugin.yaml", f"kind: model-provider\nrequires_hermes: '{primary}'\n"),
    )
    (directory / "plugin.yml").write_text(
        f"kind: model-provider\nrequires_hermes: '{secondary}'\n", encoding="utf-8"
    )
    assert (providers.get_provider_profile("compat-probe") is not None) is allowed
    assert marker.exists() is allowed


def test_unknown_release_identity_keeps_the_existing_permissive_policy(
    home, monkeypatch
):
    monkeypatch.setattr(plugins_manifest, "running_hermes_version", lambda: "unknown")
    marker = write_plugin(
        home / "plugins/model-providers/compat-probe",
        (
            "plugin.yaml",
            "kind: model-provider\nrequires_hermes: '>0.22.6'\nfuture_field: ok\n",
        ),
    )
    assert providers.get_provider_profile("compat-probe") is not None
    assert marker.exists()


@pytest.mark.parametrize("reader", ["lookup", "descriptive"])
@pytest.mark.parametrize(("spec", "allowed"), [(">0.22.6", False), ("==0.22.6", True)])
def test_bundled_directory_uses_the_same_gate(home, monkeypatch, reader, spec, allowed):
    bundled = home / "bundled"
    directory = bundled / ("compat_" + home.name)
    marker = write_plugin(
        directory,
        (
            "plugin.yaml",
            f"name: compat-probe\nkind: model-provider\nrequires_hermes: '{spec}'\n",
        ),
    )
    monkeypatch.setattr(providers, "_BUNDLED_PLUGINS_DIR", bundled)
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_SOURCES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", False)
    if reader == "lookup":
        selected = providers.get_provider_profile("compat-probe") is not None
    else:
        selected = any(
            profile.name == "compat-probe" for profile in providers.bundled_profiles()
        )
    assert selected is allowed
    assert marker.exists() is allowed
