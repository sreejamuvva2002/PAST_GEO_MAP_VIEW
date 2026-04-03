from backend.query_planner import QueryPlanner


def test_county_filter_extracts_only_known_counties() -> None:
    planner = QueryPlanner(
        company_names=["Kia Georgia Inc."],
        county_names=["Fulton", "Cobb", "Troup"],
    )

    plan = planner.plan("Map manufacturers in Fulton County and Cobb County.")

    assert plan["classification"] == "HYBRID_QUERY"
    assert plan["hints"]["counties"] == ["Fulton", "Cobb"]


def test_disruption_query_extracts_company_and_radius() -> None:
    planner = QueryPlanner(
        company_names=["Kia Georgia Inc."],
        county_names=["Fulton", "Cobb", "Troup"],
    )

    plan = planner.plan(
        "If Kia Georgia Inc. is disrupted, what nearby alternative suppliers exist within 150 miles?"
    )

    assert plan["classification"] == "DISRUPTION_ALTERNATIVES_QUERY"
    assert plan["hints"]["company_name"] == "Kia Georgia Inc."
    assert plan["hints"]["radius_km"] == 241.401
    assert plan["hints"]["analysis_intent"] == "disruption_alternatives"
