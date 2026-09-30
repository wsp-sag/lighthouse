"""Contract checks use small explicit data, including deliberately corrupt outputs."""

import copy
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "model_ci", Path(__file__).parents[1] / "scripts/model_ci.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


@pytest.fixture
def population():
    h = pd.DataFrame({"hhsize": [1]}, index=pd.Index([10], name="household_id"))
    p = pd.DataFrame(
        {"household_id": [10], "age": [40], "pemploy": [1]},
        index=pd.Index([20], name="person_id"),
    )
    return {"households": h, "persons": p}


@pytest.fixture
def outputs(population):
    tables = copy.deepcopy(population)
    tables["tours"] = pd.DataFrame(
        {
            "person_id": [20],
            "household_id": [10],
            "origin": [1],
            "destination": [2],
            "tour_mode": ["WALK"],
            "start": [8],
            "end": [17],
        },
        index=[30],
    )
    tables["trips"] = pd.DataFrame(
        {
            "person_id": [20, 20],
            "household_id": [10, 10],
            "tour_id": [30, 30],
            "origin": [1, 2],
            "destination": [2, 1],
            "trip_mode": ["WALK", "WALK"],
            "depart": [8, 17],
            "outbound": [True, False],
            "trip_num": [1, 1],
            "trip_count": [1, 1],
        },
        index=[40, 41],
    )
    return tables


def test_valid_outputs(population, outputs):
    ci.validate_outputs(population, outputs, [1, 2])


@pytest.fixture
def multistop_outputs(outputs):
    # Two outbound trips and three inbound trips; numbering restarts on return.
    trips = outputs["trips"].iloc[[0, 0, 1, 1, 1]].copy()
    trips.index = [40, 41, 42, 43, 44]
    trips["origin"] = [1, 3, 2, 4, 3]
    trips["destination"] = [3, 2, 4, 3, 1]
    trips["trip_num"] = [1, 2, 1, 2, 3]
    trips["trip_count"] = [2, 2, 3, 3, 3]
    trips["depart"] = [8, 9, 17, 18, 19]
    outputs["trips"] = trips
    return outputs


def test_valid_multistop_round_trip(population, multistop_outputs):
    # Neither table order nor trip IDs define travel order.
    multistop_outputs["trips"] = multistop_outputs["trips"].iloc[[3, 1, 4, 0, 2]]
    ci.validate_outputs(population, multistop_outputs, [1, 2, 3, 4])


def test_allows_independently_scheduled_directions(population, multistop_outputs):
    # ActivitySim 1.5.1 can produce overlapping leg schedules. Preserve the
    # per-leg time contract rather than turning this existing behavior blocking.
    multistop_outputs["trips"].loc[[42, 43, 44], "depart"] = [8, 9, 10]
    ci.validate_outputs(population, multistop_outputs, [1, 2, 3, 4])


@pytest.mark.parametrize(
    "row,column,value,message",
    [
        (42, "trip_num", 3, "broken sequence"),
        (44, "trip_num", 2, "broken sequence"),
        (42, "trip_count", 5, "incorrect trip_count"),
        (40, "origin", 3, "incorrect tour endpoints"),
        (41, "destination", 3, "incorrect tour endpoints"),
        (42, "origin", 3, "incorrect tour endpoints"),
        (44, "destination", 3, "incorrect tour endpoints"),
        (43, "origin", 1, "disconnected path"),
        (43, "depart", 16, "departure order"),
    ],
)
def test_rejects_corrupt_round_trip(
    population, multistop_outputs, row, column, value, message
):
    multistop_outputs["trips"].loc[row, column] = value
    with pytest.raises(ValueError, match=message):
        ci.validate_outputs(population, multistop_outputs, [1, 2, 3, 4])


@pytest.mark.parametrize("outbound", [True, False])
def test_rejects_missing_direction(population, outputs, outbound):
    outputs["trips"] = outputs["trips"].loc[outputs["trips"].outbound == outbound]
    with pytest.raises(ValueError, match="missing tour direction"):
        ci.validate_outputs(population, outputs, [1, 2])


@pytest.mark.parametrize(
    "table,column,value,message",
    [
        ("persons", "household_id", 999, "unknown household"),
        ("households", "hhsize", 2, "hhsize"),
        ("tours", "person_id", 999, "unknown person"),
        ("trips", "tour_id", 999, "unknown tour"),
        ("trips", "destination", 999, "invalid destination"),
        ("tours", "start", 23, "invalid times"),
        ("trips", "depart", 25, "invalid departure"),
        ("trips", "trip_num", 2, "broken sequence"),
        ("trips", "trip_count", 2, "incorrect trip_count"),
        ("trips", "trip_mode", None, "missing mode"),
    ],
)
def test_rejects_corrupt_outputs(population, outputs, table, column, value, message):
    outputs[table].iloc[0, outputs[table].columns.get_loc(column)] = value
    with pytest.raises(ValueError, match=message):
        ci.validate_outputs(population, outputs, [1, 2])


def test_rejects_population_loss(population, outputs):
    population["households"].loc[99] = [1]
    with pytest.raises(ValueError, match="population changed"):
        ci.validate_outputs(population, outputs, [1, 2])


def test_rejects_duplicate_ids(population, outputs):
    outputs["trips"].index = [40, 40]
    with pytest.raises(ValueError, match="invalid IDs"):
        ci.validate_outputs(population, outputs, [1, 2])


def test_distribution_changes_are_advisory():
    report = ci.advisory(
        {"trips.trip_mode": {"WALK": 1.0}}, {"trips.trip_mode": {"CAR": 1.0}}
    )
    assert "+100.00 pp" in report
    assert "-100.00 pp" in report


def test_backend_comparison_matches_decoded_choices(outputs):
    reference = copy.deepcopy(outputs)
    reference["tours"]["mode_choice_logsum"] = 1.0
    outputs["tours"]["mode_choice_logsum"] = 1.000001
    outputs["persons"]["_original_zone_id"] = 123
    report = ci.compare_outputs(outputs, reference)
    assert report["tours.mode_choice_logsum"] == pytest.approx(0.000001)


@pytest.mark.parametrize(
    "table,column,value",
    [
        ("trips", "destination", 99),
        ("trips", "trip_mode", "CAR"),
        ("persons", "age", 41),
        ("tours", "start", 8.000001),
    ],
)
def test_backend_comparison_rejects_changed_decisions(outputs, table, column, value):
    changed = copy.deepcopy(outputs)
    changed[table][column] = value
    with pytest.raises(ValueError, match="Backend mismatch"):
        ci.compare_outputs(changed, outputs)


def test_backend_comparison_rejects_large_logsum_difference(outputs):
    outputs["tours"]["mode_choice_logsum"] = 1.0
    changed = copy.deepcopy(outputs)
    changed["tours"]["mode_choice_logsum"] = 1.1
    with pytest.raises(ValueError, match="mode_choice_logsum"):
        ci.compare_outputs(changed, outputs)


@pytest.mark.parametrize("single_process", [True, False])
@pytest.mark.parametrize("sharrow", ["off", "require"])
def test_ci_execution_overlays(tmp_path, single_process, sharrow):
    paths = ci.model_configs(tmp_path, single_process, sharrow)
    assert (ci.ROOT / "model/configs_sh" in paths) == (sharrow == "require")
    assert paths[0] == tmp_path / "config"
    runtime = yaml.safe_load((paths[0] / "settings.yaml").read_text())
    assert (runtime.get("multiprocess") is False) == single_process
    assert runtime["sharrow_cache_dir"] == str(tmp_path / "sharrow_cache")
    assert paths[-1] == ci.ROOT / "model/configs"


def test_backend_comparison_requires_matching_provenance():
    off = {
        "seed": 0,
        "sample_households": 2000,
        "single_process": False,
        "packages": {"activitysim": "1.5.1"},
        "returncode": 0,
        "sharrow": "off",
        "sha256": {"model/data/land_use.csv": "same"},
    }
    on = copy.deepcopy(off)
    on["sharrow"] = "require"
    on["sha256"]["model/configs_sh/settings.yaml"] = "overlay"
    ci.validate_comparison_metadata(on, off)
    on["sha256"]["model/data/land_use.csv"] = "changed"
    with pytest.raises(ValueError, match="changed model/data/land_use.csv"):
        ci.validate_comparison_metadata(on, off)
    on["sha256"]["model/data/land_use.csv"] = "same"
    on["seed"] = 1
    with pytest.raises(ValueError, match="different seed"):
        ci.validate_comparison_metadata(on, off)
