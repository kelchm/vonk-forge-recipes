from __future__ import annotations

import json
import os
import runpy
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest
from vonk_forge_contracts import ModelDefinition, RecipeDefinition, content_sha256

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/canonical-synthetic-canary"
TOOL = runpy.run_path(str(ROOT / "tools/build-catalog-index"))
PACKAGE = FIXTURE / "package/canonical-synthetic-canary.tar.gz"


def _documents() -> tuple[dict[str, object], dict[str, object], dict[str, dict[str, object]]]:
    model = json.loads((FIXTURE / "model.json").read_text(encoding="utf-8"))
    model = ModelDefinition.model_validate(model).model_dump(
        mode="json", exclude_unset=False, exclude_none=False
    )
    recipe = json.loads((FIXTURE / "recipe.json").read_text(encoding="utf-8"))
    key = f"{model['identity']['publisher']}/{model['identity']['slug']}"
    return model, recipe, {key: model}


def test_canonical_canary_is_schema2_and_excluded_from_public_catalog() -> None:
    model_document, recipe_document, _ = _documents()
    model = ModelDefinition.model_validate(model_document)
    recipe = RecipeDefinition.model_validate(recipe_document)

    assert model.identity.publisher == recipe.identity.publisher == "vonk-forge-test"
    assert recipe.identity.slug == "canonical-synthetic-canary"
    assert recipe_document["models"][0]["model"]["content_sha256"] == content_sha256(model)
    assert recipe.execution.mode == "build"
    model_file = model.files[0]
    assert model_file.path == "configuration.json"
    assert (
        f"{model.source.repository}/resolve/{model.source.revision}/{model_file.path}"
    ) == (
        "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/"
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/configuration.json"
    )
    assert model_file.sha256 == "2d4464e2ead06bc9bc718c781309ad1e7baded626d66e8dcdc8b469ba185faf0"
    assert model_file.size_bytes == 51
    assert recipe.execution.build.base_image.platform == "linux/arm64"
    assert recipe.execution.build.base_image.digest != "0" * 64
    assert recipe.execution.build.base_image.digest != "f" * 64

    public_index = json.loads((ROOT / "catalog-index.json").read_text(encoding="utf-8"))
    public_recipes = {row["document"]["identity"]["slug"] for row in public_index["recipes"]}
    public_models = {row["document"]["identity"]["slug"] for row in public_index["catalog_entities"]}
    assert recipe.identity.slug not in public_recipes
    assert model.identity.slug not in public_models
    fixture_index = json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))
    assert fixture_index["recipes"][0]["release"] == fixture_index["recipes"][0]["document"]["release"]
    assert fixture_index["recipes"][0]["release"]["version"] == "1.0.1"
    assert fixture_index["schema_version"] == 2
    assert fixture_index["kind"] == "recipe-library-index"
    assert len(fixture_index["recipes"]) == 1
    assert len(fixture_index["catalog_entities"]) == 1
    assert fixture_index["recipes"][0]["document"]["identity"]["slug"] == recipe.identity.slug
    assert fixture_index["recipes"][0]["package"]["path"] == (
        "tests/fixtures/canonical-synthetic-canary/package/canonical-synthetic-canary.tar.gz"
    )


def test_canonical_canary_huggingface_source_is_accepted_by_model_cache() -> None:
    platform_root = Path(os.environ.get("VONK_FORGE_PLATFORM_ROOT", "/opt/vonk-forge"))
    control_source = platform_root / "control/src"
    if not control_source.is_dir():
        pytest.skip("authoritative platform checkout is unavailable")
    sys.path.insert(0, str(control_source))
    try:
        try:
            from vonk_control.model_cache import _source_for_catalog_artifact
        except ImportError:
            pytest.skip("platform ModelCache dependencies are unavailable")
        model_document, _, _ = _documents()
        model = ModelDefinition.model_validate(model_document)
        model_file = model.files[0]
        source, revision = _source_for_catalog_artifact(
            {
                "kind": "huggingface.file",
                "repository": model.source.repository,
                "revision": model.source.revision,
                "path": model_file.path,
            }
        )
        assert source == (
            "https://huggingface.co/Qwen/Qwen3.6-27B/resolve/"
            "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/configuration.json"
        )
        assert revision == model.source.revision
    finally:
        sys.path.remove(str(control_source))


def test_canonical_canary_package_has_exact_source_and_model_closure() -> None:
    model_document, recipe_document, entities = _documents()
    payload, metadata = TOOL["recipe_package"](
        recipe_document,
        recipe_path=FIXTURE / "recipe.json",
        entity_documents=entities,
    )
    assert payload == PACKAGE.read_bytes()
    fixture_index = json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))
    assert metadata["sha256"] == fixture_index["recipes"][0]["package"]["sha256"]
    TOOL["validate_recipe_archive"](payload, recipe_document, entities)

    with tarfile.open(fileobj=BytesIO(payload), mode="r:gz") as archive:
        names = set(archive.getnames())
        recipe_name = f"models/{model_document['identity']['slug']}.json"
        assert names == {
            "manifest.json",
            "recipe.json",
            recipe_name,
            "tests/fixtures/canonical-synthetic-canary/context/Dockerfile",
            "tests/fixtures/canonical-synthetic-canary/context/server.py",
        }
        dockerfile = archive.extractfile(
            "tests/fixtures/canonical-synthetic-canary/context/Dockerfile"
        ).read().decode()
        assert "USER 10001:10001" in dockerfile
        assert "@sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3" in dockerfile


@pytest.mark.parametrize("change,accepted", [
    ({}, True), ({"stream": None}, True), ({"stream": True}, False),
    ({"stream": 0}, False), ({"model": "wrong"}, False),
    ({"max_tokens": 17}, False), ({"messages": []}, False),
])
def test_canonical_canary_server_behaves_without_gpu(change, accepted) -> None:
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", 0))
        except PermissionError:
            pytest.skip("local sockets are unavailable in this sandbox")
        port = listener.getsockname()[1]
    environment = {**os.environ, "VONK_LISTEN_HOST": "127.0.0.1", "VONK_LISTEN_PORT": str(port)}
    process = subprocess.Popen(
        [sys.executable, str(FIXTURE / "context/server.py")],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=0.2) as response:
                    assert json.load(response) == json.loads((FIXTURE / "expected.json").read_text())["health"]
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        body = json.loads((FIXTURE / "expected.json").read_text())["request"]
        body.update(change)
        if change.get("stream", "present") is None:
            del body["stream"]
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if accepted:
            with urllib.request.urlopen(request, timeout=2) as response:
                assert json.load(response) == json.loads((FIXTURE / "expected.json").read_text())["response"]
        else:
            with pytest.raises(urllib.error.HTTPError) as rejected:
                urllib.request.urlopen(request, timeout=2)
            assert rejected.value.code == 400
    finally:
        process.terminate()
        process.wait(timeout=5)
