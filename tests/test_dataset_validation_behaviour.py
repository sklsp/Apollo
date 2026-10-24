"""Behaviour tests for app/services/dataset_validation.py."""

from __future__ import annotations

import pytest


def _png(width: int, height: int, color=(200, 30, 90)) -> bytes:
    """A solid-color RGB PNG of the given size."""
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (width, height), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def _gradient_png(width: int, height: int) -> bytes:
    """A vertical grayscale gradient - structured aHash, not low-entropy."""
    from io import BytesIO

    from PIL import Image

    img = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _entry(image_id: str, caption="a proper caption", filename=None) -> dict:
    entry = {"id": image_id, "caption": caption}
    if filename is not None:
        entry["filename"] = filename
    return entry


# ---------- ImageFinding / DatasetReport --------------------------------------


class TestImageFindingToDict:
    def test_to_dict_returns_all_fields(self):
        from app.services.dataset_validation import ImageFinding

        finding = ImageFinding("img1", "a.png", "duplicate", "identical to img0")

        assert finding.to_dict() == {
            "image_id": "img1",
            "filename": "a.png",
            "kind": "duplicate",
            "detail": "identical to img0",
        }


class TestDatasetReport:
    def test_issue_count_tracks_findings(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding

        report = DatasetReport(total_images=3)
        assert report.issue_count == 0
        report.findings.append(ImageFinding("a", "a.png", "invalid"))
        report.findings.append(ImageFinding("b", "b.png", "duplicate"))

        assert report.issue_count == 2

    def test_to_dict_aggregates_finding_kinds(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding

        report = DatasetReport(total_images=4, valid_images=3, captioned=1)
        report.average_resolution = (512, 512)
        report.score = 78
        for image_id, kind in [
            ("a", "missing_caption"),
            ("b", "duplicate"),
            ("c", "near_duplicate"),
            ("d", "near_duplicate_possible"),
            ("e", "extreme_resolution"),
        ]:
            report.findings.append(ImageFinding(image_id, f"{image_id}.png", kind))

        data = report.to_dict()

        assert data["total_images"] == 4
        assert data["valid_images"] == 3
        assert data["invalid_images"] == 1
        assert data["captioned"] == 1
        assert data["missing_captions"] == 1
        assert data["duplicates"] == 1
        assert data["near_duplicates"] == 1
        assert data["possible_duplicates"] == 1
        assert data["extreme_resolutions"] == 1
        assert data["average_resolution"] == [512, 512]
        assert data["issue_count"] == 5
        assert data["score"] == 78
        assert len(data["findings"]) == 5

    def test_to_dict_average_resolution_none_when_unset(self):
        from app.services.dataset_validation import DatasetReport

        assert DatasetReport().to_dict()["average_resolution"] is None


# ---------- classify_pair -----------------------------------------------------


class TestClassifyPair:
    def test_identical_flat_hashes_need_matching_color(self):
        from app.services.dataset_validation import classify_pair

        flat = "0" * 16
        # Far apart mean colors -> not the same picture.
        assert (
            classify_pair(flat, flat, color_a=(255, 0, 0), color_b=(0, 0, 255))
            == "unique"
        )
        # Close mean colors -> high-confidence near-duplicate.
        assert (
            classify_pair(flat, flat, color_a=(100, 100, 100), color_b=(105, 98, 102))
            == "near_duplicate_high"
        )

    def test_identical_hash_without_colors_is_high(self):
        from app.services.dataset_validation import classify_pair

        assert (
            classify_pair("5555555555555555", "5555555555555555")
            == "near_duplicate_high"
        )

    def test_identical_structured_hash_ignores_color(self):
        from app.services.dataset_validation import classify_pair

        # Not low-entropy, so the color corroboration does not apply.
        assert (
            classify_pair(
                "5555555555555555",
                "5555555555555555",
                color_a=(255, 0, 0),
                color_b=(0, 0, 255),
            )
            == "near_duplicate_high"
        )

    def test_distance_above_close_band_is_unique(self):
        from app.services.dataset_validation import classify_pair

        # Hamming distance 7 > NEAR_HASH_CLOSE (6).
        assert classify_pair("5555555555555555", "555555555555552a") == "unique"

    def test_close_distance_with_matching_aspect_is_high(self):
        from app.services.dataset_validation import classify_pair

        # Distance 3, same aspect ratio -> corroborated.
        assert (
            classify_pair(
                "5555555555555555",
                "5555555555555552",
                dims_a=(400, 300),
                dims_b=(800, 600),
            )
            == "near_duplicate_high"
        )

    def test_close_distance_at_band_edge_is_not_unique(self):
        from app.services.dataset_validation import classify_pair

        # Distance exactly 6 (band edge) with no signals -> still high.
        assert (
            classify_pair("5555555555555555", "555555555555556a")
            == "near_duplicate_high"
        )

    def test_close_distance_with_mismatched_aspect_is_possible(self):
        from app.services.dataset_validation import classify_pair

        assert (
            classify_pair(
                "5555555555555555",
                "5555555555555552",
                dims_a=(400, 300),
                dims_b=(800, 400),
            )
            == "near_duplicate_possible"
        )

    def test_close_distance_without_signals_is_high(self):
        from app.services.dataset_validation import classify_pair

        # Nothing to corroborate against -> nothing can disagree.
        assert (
            classify_pair("5555555555555555", "5555555555555552")
            == "near_duplicate_high"
        )

    def test_close_flat_hashes_with_far_colors_are_possible(self):
        from app.services.dataset_validation import classify_pair

        assert (
            classify_pair(
                "0" * 16,
                "0000000000000007",
                color_a=(255, 0, 0),
                color_b=(0, 255, 0),
            )
            == "near_duplicate_possible"
        )

    def test_close_flat_hashes_with_matching_colors_are_high(self):
        from app.services.dataset_validation import classify_pair

        assert (
            classify_pair(
                "0" * 16,
                "0000000000000007",
                color_a=(10, 10, 10),
                color_b=(14, 9, 11),
            )
            == "near_duplicate_high"
        )



# ---------- pure helpers ------------------------------------------------------


class TestHamming:
    def test_identical_hashes_distance_zero(self):
        from app.services.dataset_validation import _hamming

        assert _hamming("5555555555555555", "5555555555555555") == 0

    def test_opposite_hashes_distance_sixty_four(self):
        from app.services.dataset_validation import _hamming

        assert _hamming("0" * 16, "f" * 16) == 64


class TestColorDistance:
    def test_uses_max_channel_difference(self):
        from app.services.dataset_validation import _color_distance

        assert _color_distance((10, 200, 30), (15, 205, 90)) == 60
        assert _color_distance((100, 100, 100), (100, 100, 100)) == 0


class TestIsLowEntropy:
    @pytest.mark.parametrize(
        "hash_hex",
        [
            "0" * 16,          # zero ones
            "f" * 16,          # sixty-four ones
            "000000000000001b",  # exactly four ones (boundary)
            "ffffffffffffffe4",  # exactly sixty ones (boundary)
        ],
    )
    def test_flat_hashes_are_low_entropy(self, hash_hex):
        from app.services.dataset_validation import _is_low_entropy

        assert _is_low_entropy(hash_hex) is True

    @pytest.mark.parametrize(
        "hash_hex",
        [
            "000000000000002f",  # five ones - just above the boundary
            "5555555555555555",  # thirty-two ones
        ],
    )
    def test_structured_hashes_are_not_low_entropy(self, hash_hex):
        from app.services.dataset_validation import _is_low_entropy

        assert _is_low_entropy(hash_hex) is False


class TestSameAspectRatio:
    def test_same_ratio_within_tolerance(self):
        from app.services.dataset_validation import _same_aspect_ratio

        assert _same_aspect_ratio(400, 300, 800, 600) is True
        # 5% tolerance: 1.333 vs ~1.332 stays inside the band.
        assert _same_aspect_ratio(400, 300, 800, 601) is True

    def test_different_ratios_rejected(self):
        from app.services.dataset_validation import _same_aspect_ratio

        assert _same_aspect_ratio(400, 300, 600, 400) is False

    def test_zero_width_is_never_the_same_shape(self):
        from app.services.dataset_validation import _same_aspect_ratio

        assert _same_aspect_ratio(0, 100, 0, 200) is False


class TestFilenameOf:
    def test_returns_filename_for_matching_entry(self):
        from app.services.dataset_validation import _filename_of

        entries = [{"id": "a", "filename": "a.png"}, {"id": "b"}]

        assert _filename_of(entries, "a") == "a.png"
        # Entry without a filename key falls back to its id.
        assert _filename_of(entries, "b") == "b"

    def test_falls_back_to_image_id_when_no_entry_matches(self):
        from app.services.dataset_validation import _filename_of

        assert _filename_of([{"id": "a", "filename": "a.png"}], "zzz") == "zzz"
        assert _filename_of([], "solo") == "solo"


# ---------- validate_dataset --------------------------------------------------


class TestValidateDatasetClean:
    def test_clean_dataset_scores_hundred(self):
        from app.services.dataset_validation import validate_dataset

        images = {
            "a": _png(300, 300, color=(255, 0, 0)),
            "b": _png(300, 300, color=(0, 0, 255)),
        }
        entries = [
            _entry("a", caption="red square", filename="a.png"),
            _entry("b", caption="blue square", filename="b.png"),
        ]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        assert report.total_images == 2
        assert report.valid_images == 2
        assert report.captioned == 2
        assert report.findings == []
        assert report.score == 100
        # Both images are 300x300.
        assert report.average_resolution == (300, 300)

    def test_empty_dataset_reports_nothing(self):
        from app.services.dataset_validation import validate_dataset

        report = validate_dataset([], lambda image_id: b"")

        assert report.total_images == 0
        assert report.valid_images == 0
        assert report.average_resolution is None
        assert report.findings == []
        assert report.score == 100


class TestValidateDatasetCaptions:
    def test_missing_empty_and_suspicious_captions_flagged(self):
        from app.services.dataset_validation import validate_dataset

        # Distinct colors keep every image unique (no duplicate findings).
        images = {
            "a": _png(300, 300, color=(255, 0, 0)),
            "b": _png(300, 300, color=(0, 0, 255)),
            "c": _png(300, 300, color=(0, 180, 0)),
            "d": _png(300, 300, color=(240, 200, 0)),
        }
        entries = [
            {"id": "a"},  # no caption key at all
            _entry("b", caption="   "),  # whitespace only
            _entry("c", caption="ab"),  # too short
            _entry(
                "d",
                caption="Sorry, I cannot describe this image because it is not safe.",
            ),
        ]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        by_kind = {f.kind for f in report.findings}
        assert {"missing_caption", "empty_caption", "suspicious_caption"} <= by_kind
        details = {f.image_id: f.detail for f in report.findings}
        assert details["a"] == ""  # missing caption carries no detail
        assert details["b"] == ""
        assert details["c"] == "ab"
        assert details["d"].startswith("Sorry, I cannot describe")
        assert report.captioned == 0

    @pytest.mark.parametrize(
        "caption",
        [
            "As an AI language model, I ...",
            "I cannot generate this image.",
            "I can't help you with that.",
            "Sorry, something went wrong.",
            "lorem ipsum dolor sit amet",
            "the background is undefined here",
            "caption: null, end of line",
            "[error] failed to caption",
        ],
    )
    def test_each_suspicious_marker_is_detected(self, caption):
        from app.services.dataset_validation import validate_dataset

        images = {"a": _png(300, 300)}
        report = validate_dataset([_entry("a", caption=caption)], lambda i: images[i])

        assert [f.kind for f in report.findings] == ["suspicious_caption"]


class TestValidateDatasetDecoding:
    def test_garbage_bytes_flagged_invalid(self):
        from app.services.dataset_validation import validate_dataset

        entries = [_entry("a", filename="a.png")]

        report = validate_dataset(entries, lambda image_id: b"not an image")

        assert report.valid_images == 0
        invalid = [f for f in report.findings if f.kind == "invalid"]
        assert len(invalid) == 1
        assert invalid[0].image_id == "a"
        assert invalid[0].filename == "a.png"

    def test_read_bytes_exception_flagged_invalid(self):
        from app.services.dataset_validation import validate_dataset

        entries = [_entry("a", filename="missing.png")]

        def _boom(image_id):
            raise FileNotFoundError(f"no such file: {image_id}")

        report = validate_dataset(entries, _boom)

        assert [f.kind for f in report.findings] == ["invalid"]


class TestValidateDatasetResolution:
    def test_below_minimum_flagged(self):
        from app.services.dataset_validation import validate_dataset

        images = {"a": _png(100, 300)}
        report = validate_dataset([_entry("a")], lambda i: images[i])

        extreme = [f for f in report.findings if f.kind == "extreme_resolution"]
        assert len(extreme) == 1
        assert "below" in extreme[0].detail and "256" in extreme[0].detail

    def test_above_maximum_flagged(self):
        from app.services.dataset_validation import validate_dataset

        images = {"a": _png(9000, 300)}
        report = validate_dataset([_entry("a")], lambda i: images[i])

        extreme = [f for f in report.findings if f.kind == "extreme_resolution"]
        assert len(extreme) == 1
        assert "above" in extreme[0].detail and "8192" in extreme[0].detail


class TestValidateDatasetDuplicates:
    def test_identical_bytes_flagged_duplicate(self):
        from app.services.dataset_validation import validate_dataset

        blob = _png(300, 300)
        images = {"a": blob, "b": blob}
        entries = [_entry("a"), _entry("b")]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        dups = [f for f in report.findings if f.kind == "duplicate"]
        assert len(dups) == 1
        assert dups[0].image_id == "b"
        assert dups[0].detail == "identical to a"

    def test_same_structure_different_size_flagged_near_duplicate(self):
        from app.services.dataset_validation import validate_dataset

        images = {
            "a": _gradient_png(260, 260),
            "b": _gradient_png(400, 400),
        }
        entries = [_entry("a"), _entry("b")]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        near = [f for f in report.findings if f.kind == "near_duplicate"]
        assert len(near) == 1
        assert near[0].image_id == "b"
        assert near[0].detail == "near duplicate high vs a"

    def test_possible_verdict_flagged_with_lower_severity(self, monkeypatch):
        import app.services.dataset_validation as dv
        from app.services.dataset_validation import validate_dataset

        images = {k: _png(300, 300) for k in ("a", "b", "c")}
        entries = [_entry("a"), _entry("b"), _entry("c")]

        monkeypatch.setattr(dv, "classify_pair", lambda *a, **kw: "near_duplicate_possible")

        report = validate_dataset(entries, lambda image_id: images[image_id])

        near = [f for f in report.findings if f.kind == "near_duplicate_possible"]
        # Pairs (a,b), (a,c), (b,c) -> later ids b, c, c.
        assert sorted(f.image_id for f in near) == ["b", "c", "c"]
        assert all("possible" in f.detail for f in near)

    def test_entry_without_filename_falls_back_to_image_id(self):
        from app.services.dataset_validation import validate_dataset

        blob = _png(300, 300)
        images = {"a": blob, "b": blob}
        entries = [{"id": "a"}, {"id": "b"}]  # no filename keys anywhere

        report = validate_dataset(entries, lambda image_id: images[image_id])

        dup = next(f for f in report.findings if f.kind == "duplicate")
        assert dup.filename == "b"

    def test_repeated_id_pair_produces_single_finding(self, monkeypatch):
        import app.services.dataset_validation as dv
        from app.services.dataset_validation import validate_dataset

        # First and third entries share id "a", so the (a, b) id-pair is seen
        # twice by index. Every pair classifies as a near duplicate; the
        # repeated id-pair must still be flagged only once.
        red = _png(300, 300, color=(255, 0, 0))
        blue = _png(300, 300, color=(0, 0, 255))
        entries = [_entry("a"), _entry("b"), _entry("a")]

        monkeypatch.setattr(dv, "classify_pair", lambda *a, **kw: "near_duplicate_high")

        report = validate_dataset(entries, lambda image_id: {"a": red, "b": blue}[image_id])

        near = [f for f in report.findings if f.kind == "near_duplicate"]
        assert [(f.image_id, f.detail) for f in near] == [
            ("b", "near duplicate high vs a"),
            ("a", "near duplicate high vs a"),
        ]


class TestValidateDatasetDimensions:
    def test_mixed_dimensions_flagged_once(self):
        from app.services.dataset_validation import validate_dataset

        images = {
            "a": _png(300, 300, color=(255, 0, 0)),
            "b": _png(400, 400, color=(0, 0, 255)),
        }
        entries = [_entry("a"), _entry("b")]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        dim_findings = [f for f in report.findings if f.kind == "dimension_inconsistency"]
        assert len(dim_findings) == 1
        finding = dim_findings[0]
        assert finding.image_id == "*"
        assert finding.filename == "(dataset)"
        assert "2 distinct dimensions" in finding.detail

    def test_uniform_dimensions_not_flagged(self):
        from app.services.dataset_validation import validate_dataset

        images = {
            "a": _png(300, 300, color=(255, 0, 0)),
            "b": _png(300, 300, color=(0, 0, 255)),
        }
        entries = [_entry("a"), _entry("b")]

        report = validate_dataset(entries, lambda image_id: images[image_id])

        assert not [f for f in report.findings if f.kind == "dimension_inconsistency"]


# ---------- scoring -----------------------------------------------------------


class TestScore:
    def test_clean_report_scores_hundred(self):
        from app.services.dataset_validation import DatasetReport, _score

        assert _score(DatasetReport(total_images=5, valid_images=5)) == 100

    def test_penalties_are_weighted_per_kind(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding, _score

        report = DatasetReport(total_images=4, valid_images=2)
        # 2 invalid (30) + 1 missing caption (5) + 1 extreme resolution (6) = 41.
        for image_id, kind in [
            ("a", "invalid"),
            ("b", "invalid"),
            ("c", "missing_caption"),
            ("d", "extreme_resolution"),
        ]:
            report.findings.append(ImageFinding(image_id, f"{image_id}.png", kind))

        assert _score(report) == 59

    def test_per_kind_penalty_is_capped(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding, _score

        report = DatasetReport(total_images=12)
        for i in range(12):
            report.findings.append(ImageFinding(f"i{i}", "x.png", "missing_caption"))

        # 12 * 5 would be 60; the cap limits it to 10 * 5 = 50.
        assert _score(report) == 50

    def test_score_never_goes_below_zero(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding, _score

        report = DatasetReport(total_images=10)
        for i in range(10):
            report.findings.append(ImageFinding(f"i{i}", "x.png", "invalid"))

        # 10 * 15 = 150 penalty -> clamped to zero.
        assert _score(report) == 0

    def test_unknown_kind_uses_default_weight(self):
        from app.services.dataset_validation import DatasetReport, ImageFinding, _score

        report = DatasetReport(total_images=3)
        for i in range(3):
            report.findings.append(ImageFinding(f"i{i}", "x.png", "mystery"))

        assert _score(report) == 94



