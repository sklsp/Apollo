"""Behavioural tests for app/services/prompt_service.py."""

import pytest

from app.services import prompt_service as module
from app.services.prompt_service import PromptService


class TestDefaultsAndCreate:
    def test_constructor_seeds_four_default_prompts(self):
        svc = PromptService()

        prompts = {p["name"]: p for p in svc.list_prompts()}
        assert set(prompts) == {"default", "code_helper", "document_analyzer", "summarizer"}
        # Creation order: default, code_helper, document_analyzer, summarizer.
        assert [p["id"] for p in (prompts[n] for n in ("default", "code_helper", "document_analyzer", "summarizer"))] == [
            "prompt_0",
            "prompt_1",
            "prompt_2",
            "prompt_3",
        ]

    def test_create_prompt_returns_incrementing_ids(self):
        svc = PromptService()

        new_id = svc.create_prompt("custom", "body {input}")

        assert new_id == "prompt_4"  # counter continues past the four defaults
        stored = svc.get_prompt(new_id)
        assert stored["name"] == "custom"
        assert stored["content"] == "body {input}"


class TestRetrieve:
    def test_get_prompt_missing_returns_none(self):
        assert PromptService().get_prompt("prompt_99") is None

    def test_get_prompt_by_name_found_and_missing(self):
        svc = PromptService()

        found = svc.get_prompt_by_name("summarizer")
        assert found is not None and "{input}" in found["content"]

        assert svc.get_prompt_by_name("nope") is None

    def test_list_prompts_sorted_case_insensitively(self):
        svc = PromptService()
        svc.create_prompt("Zeta", "z")
        svc.create_prompt("alpha", "a")

        names = [p["name"] for p in svc.list_prompts()]
        assert names == ["alpha", "code_helper", "default", "document_analyzer", "summarizer", "Zeta"]


class TestUpdate:
    def test_update_missing_id_returns_false(self):
        assert PromptService().update_prompt("prompt_99", name="x") is False

    @pytest.fixture()
    def clock(self, monkeypatch):
        """Deterministic _utc_now: t0, t1, t2, ... in call order."""
        ticks = iter(f"t{i}" for i in range(50))
        monkeypatch.setattr(module, "_utc_now", lambda: next(ticks))
        return ticks

    def test_update_name_and_content_bumps_updated_at(self, clock):
        svc = PromptService()  # consumes t0..t7 (4 defaults x created+updated)
        prompt_id = svc.create_prompt("old", "old body")  # t8/t9
        original_created = svc.get_prompt(prompt_id)["created_at"]

        assert svc.update_prompt(prompt_id, name="new", content="new body") is True

        updated = svc.get_prompt(prompt_id)
        assert (updated["name"], updated["content"]) == ("new", "new body")
        assert updated["created_at"] == original_created  # created_at untouched
        assert updated["updated_at"] != original_created  # bumped by the update

    def test_update_partial_fields_only_touches_given_ones(self, clock):
        svc = PromptService()
        prompt_id = svc.create_prompt("keep-name", "keep-content")

        assert svc.update_prompt(prompt_id, name="renamed") is True
        renamed = svc.get_prompt(prompt_id)
        assert (renamed["name"], renamed["content"]) == ("renamed", "keep-content")

        assert svc.update_prompt(prompt_id, content="new-body") is True
        edited = svc.get_prompt(prompt_id)
        assert (edited["name"], edited["content"]) == ("renamed", "new-body")

    def test_update_with_no_fields_still_bumps_timestamp(self, clock):
        svc = PromptService()
        prompt_id = svc.create_prompt("x", "y")
        before = svc.get_prompt(prompt_id)["updated_at"]

        assert svc.update_prompt(prompt_id) is True

        after = svc.get_prompt(prompt_id)["updated_at"]
        assert (after, before) != (before, after)


class TestDelete:
    def test_delete_existing_and_missing(self):
        svc = PromptService()
        prompt_id = svc.create_prompt("doomed", "x")

        assert svc.delete_prompt(prompt_id) is True
        assert svc.get_prompt(prompt_id) is None
        assert svc.delete_prompt(prompt_id) is False


class TestApplyTemplate:
    def test_replaces_input_placeholder(self):
        svc = PromptService()
        prompt_id = svc.create_prompt("tpl", "Do this: {input} now")

        result = svc.apply_template(prompt_id, "the task")
        assert result == "Do this: the task now"

    def test_appends_user_input_when_no_placeholder(self):
        svc = PromptService()
        prompt_id = svc.create_prompt("plain", "Just a system line")

        result = svc.apply_template(prompt_id, "user says hi")
        assert result == "Just a system line\n\nuser says hi"

    def test_missing_id_raises_keyerror(self):
        with pytest.raises(KeyError, match="prompt_99"):
            PromptService().apply_template("prompt_99", "x")
