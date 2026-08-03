"""
DATA VALIDATION - Great Expectations quality gate for the Telco Churn dataset.

Written against the Great Expectations 1.x API (Data Context -> Data Source ->
Batch Definition -> Expectation Suite -> Validation Definition). The legacy 0.x
`great_expectations.dataset.PandasDataset` helper this module used to call was
removed in GE 1.0, so it cannot be used with the pinned version.
"""

from typing import List, Tuple

import pandas as pd
import great_expectations as gx
import great_expectations.expectations as gxe


def _build_suite() -> gx.ExpectationSuite:
    """
    Assemble the Telco churn expectation suite.

    Groups the same checks the project has always run: schema, business logic,
    numeric ranges, and cross-column consistency.
    """
    suite = gx.ExpectationSuite(name="telco_churn_suite")

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    # Customer identifier must exist (required for business operations)
    suite.add_expectation(gxe.ExpectColumnToExist(column="customerID"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="customerID"))

    # Core demographic features
    for col in ["gender", "Partner", "Dependents"]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # Service features (critical for churn analysis)
    for col in ["PhoneService", "InternetService", "Contract"]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # Financial features (key churn predictors)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        suite.add_expectation(gxe.ExpectColumnToExist(column=col))

    # === BUSINESS LOGIC VALIDATION ===
    # Categorical fields must hold only their expected values (data integrity)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"])
    )
    for col in ["Partner", "Dependents", "PhoneService"]:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(column=col, value_set=["Yes", "No"])
        )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="Contract", value_set=["Month-to-month", "One year", "Two year"]
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="InternetService", value_set=["DSL", "Fiber optic", "No"]
        )
    )

    # === NUMERIC RANGE VALIDATION ===
    # Tenure is non-negative and capped at a reasonable telecom lifetime (~10 yrs)
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
    )
    # Monthly charges within a sane business range
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200)
    )
    # TotalCharges arrives as text with literal " " blanks for brand-new
    # customers, so validate_telco_data() coerces it to numeric first. GE skips
    # the resulting NaNs here, leaving only real values range-checked.
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0)
    )

    # No missing values in critical numeric features
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # === DATA CONSISTENCY CHECKS ===
    # Total charges should generally be >= monthly charges (catches entry errors).
    # mostly=0.95 tolerates new customers whose totals have not accrued yet, and
    # ignore_row_if skips the rows whose TotalCharges was blank in the raw file
    # (otherwise every brand-new customer counts as a violation).
    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="TotalCharges",
            column_B="MonthlyCharges",
            or_equal=True,
            mostly=0.95,
            ignore_row_if="either_value_is_missing",
        )
    )

    return suite


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Run the Telco churn data quality gate.

    Args:
        df: Raw (pre-preprocessing) Telco churn DataFrame.

    Returns:
        (success, failed_expectation_types) - `success` is False if any
        expectation failed; the list names the expectation types that failed so
        the caller can log them for debugging.
    """
    print("Starting data validation with Great Expectations...")

    # Validate a copy so the caller's DataFrame is untouched. TotalCharges ships
    # as text containing literal " " blanks for customers with tenure 0; coercing
    # it here lets the numeric expectations below actually evaluate. The blanks
    # become NaN, which GE skips, and preprocess_data() fills them later.
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Ephemeral context: everything lives in memory, so validation leaves no
    # great_expectations/ project directory behind in the repo.
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas(name="telco_pandas")
    data_asset = data_source.add_dataframe_asset(name="telco_churn")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_dataframe")

    suite = context.suites.add(_build_suite())
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="telco_churn_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    print("   Running complete validation suite...")
    results = validation_definition.run(batch_parameters={"dataframe": df})

    # === PROCESS RESULTS ===
    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return bool(results.success), failed_expectations
