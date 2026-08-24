import logging

import pandas as pd

from activitysim.core import estimation, config, expressions, simulate
from activitysim.core import workflow, los, tracing
from activitysim.core.configuration.base import PreprocessorSettings
from activitysim.core.configuration.logit import LogitComponentSettings

logger = logging.getLogger("activitysim")


class ConstraintCanTravelAloneSettings(LogitComponentSettings, extra="forbid"):
    """
    Settings for the can travel alone constraint.
    """

    CAN_TRAVEL_ALONE_ALT: int = 0
    """The column index number of the spec file for the can travel alone alternative."""


@workflow.step
def constraint_can_travel_alone(
    state: workflow.State,
    persons: pd.DataFrame,
    persons_merged: pd.DataFrame,
    # network_los: los.Network_LOS,
    model_settings: ConstraintCanTravelAloneSettings | None = None,
    model_settings_file_name: str = "constraint_can_travel_alone.yaml",
    trace_label: str = "constraint_can_travel_alone",
    trace_hh_id: bool = False,
) -> None:
    """
    This is the step for applying the can travel alone constraint.
    Use this step if the model needs to be a logit-based model for the can travel alone constraint.
    If it's just simple rule-based logic, you may not need to use this step.
    """

    if model_settings is None:
        model_settings = ConstraintCanTravelAloneSettings.read_settings_file(
            state.filesystem,
            model_settings_file_name,
        )

    choosers = persons_merged
    logger.info("Running %s with %d persons", trace_label, len(choosers))

    # for estimation needs
    estimator = estimation.manager.begin_estimation(
        state, "constraint_can_travel_alone"
    )

    constants = config.get_model_constants(model_settings)
    can_travel_alone_alt = model_settings.CAN_TRAVEL_ALONE_ALT

    # preprocessor
    expressions.annotate_preprocessors(
        state,
        df=persons_merged,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )

    # load model specification and coefficients
    model_spec = state.filesystem.read_model_spec(file_name=model_settings.SPEC)
    coefficients_df = state.filesystem.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(
        state, model_spec, coefficients_df, estimator
    )
    # if it is a nest logit
    nest_spec = config.get_logit_model_settings(model_settings)

    if estimator:
        estimator.write_model_settings(model_settings, model_settings_file_name)
        estimator.write_spec(model_settings)
        estimator.write_coefficients(coefficients_df)
        estimator.write_choosers(choosers)

    choices = simulate.simple_simulate(
        state,
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        trace_label=trace_label,
        trace_choice_name="can_travel_alone",
        estimator=estimator,
        compute_settings=model_settings.compute_settings,
    )

    choices = choices == can_travel_alone_alt

    if estimator:
        estimator.write_choices(choices)
        choices = estimator.get_survey_values(choices, "persons", "can_travel_alone")
        estimator.write_override_choices(choices)
        estimator.end_estimation()

    persons["can_travel_alone"] = choices.reindex(persons.index).fillna(0).astype(bool)

    state.add_table("persons", persons)

    tracing.print_summary(
        "can_travel_alone", persons.can_travel_alone, value_counts=True
    )
    logger.info("Person can travel alone summary: %s", persons.can_travel_alone.value_counts())

    if trace_hh_id:
        state.tracing.trace_df(persons, label=trace_label, warn_if_empty=True)

    # annotate result table if needed
    expressions.annotate_tables(
        state,
        locals_dict=constants,
        skims=None,
        model_settings=model_settings,
        trace_label=trace_label,
    )
