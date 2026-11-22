"""Supabase ingestion against a fake HTTP layer (no network)."""
from __future__ import annotations

import pandas as pd
import pytest

from src import ingest
from src.config import load_config


class FakeResp:
    def __init__(self, status=200, text="{}", content=b""):
        self.status_code, self.text, self._content = status, text, content

    def iter_content(self, _n):
        yield self._content


@pytest.fixture()
def fake_supabase(monkeypatch):
    store, calls = {}, []
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co/")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_x")

    def post(url, headers=None, json=None, data=None, timeout=None):
        calls.append(("POST", url, headers))
        if url.endswith("/storage/v1/bucket"):
            return FakeResp(400, '{"message":"The resource already exists"}')
        store[url.split("/object/")[1]] = data.read()
        return FakeResp(200)

    def get(url, headers=None, timeout=None, stream=None):
        calls.append(("GET", url, headers))
        return FakeResp(200, content=store[url.split("/object/")[1]])

    monkeypatch.setattr(ingest.requests, "post", post)
    monkeypatch.setattr(ingest.requests, "get", get)
    return store, calls


def test_supabase_roundtrip(tmp_path, fake_supabase):
    store, calls = fake_supabase
    src = tmp_path / "clean.csv"
    pd.DataFrame({"is_canceled": [0, 1], "lead_time": [5, 300]}).to_csv(src, index=False)
    cfg = load_config(overrides=["data.source=supabase", f"data.clean_path={src}"])

    ingest.upload(cfg)
    assert "hotel-data/hotel_bookings_clean.csv" in store
    src.unlink()
    df = ingest.load_dataset(cfg)
    assert df["lead_time"].tolist() == [5, 300]

    # new-style secret keys go in `apikey` only; trailing slash in URL is handled
    _, url, headers = calls[-1]
    assert url == "https://abc.supabase.co/storage/v1/object/hotel-data/hotel_bookings_clean.csv"
    assert headers == {"apikey": "sb_secret_x"}


def test_missing_credentials_message(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    with pytest.raises(SystemExit, match="SUPABASE_URL"):
        ingest._supabase_creds()


def test_rest_v1_suffix_is_stripped(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co/rest/v1/")
    monkeypatch.setenv("SUPABASE_KEY", "sb_secret_x")
    url, _ = ingest._supabase_creds()
    assert url == "https://abc.supabase.co"
