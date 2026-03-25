"""
EduShield Uganda - Intervention Recommender
Recommends evidence-based interventions based on student risk factors.

Sources:
- Uganda National Panel Survey (UNPS) 2019/20
- Uganda Demographic and Health Survey (UDHS) 2022
- Uganda National Household Survey (UNHS) 2019/20
- UNESCO Institute for Statistics (UIS) – Education indicators
"""


REGION_NAMES = {
    1: "Central",
    2: "Eastern",
    3: "Northern",
    4: "Western",
}

DISTRICT_SAMPLES = {
    1: ["Kampala", "Wakiso", "Mukono", "Mpigi", "Luwero"],
    2: ["Jinja", "Iganga", "Mbale", "Tororo", "Soroti"],
    3: ["Gulu", "Lira", "Arua", "Kitgum", "Adjumani"],
    4: ["Mbarara", "Kabale", "Kasese", "Fort Portal", "Bushenyi"],
}

LOW_WELFARE_THRESHOLD = 52_100.0
LARGE_HOUSEHOLD_THRESHOLD = 6

# ---------------------------------------------------------------------------
# Supplementary national-level statistics (hardcoded from official reports)
# These provide context even when survey microdata is limited to UNPS.
# ---------------------------------------------------------------------------

# UDHS 2022 - Key child indicators for children ≤10
UDHS_2022 = {
    "total_fertility_rate": 5.2,
    "infant_mortality_per_1000": 43,
    "under5_mortality_per_1000": 52,
    "stunting_pct": 26.3,          # % of children under 5 who are stunted
    "wasting_pct": 3.5,            # % of children under 5 who are wasted
    "underweight_pct": 10.4,       # % of children under 5 who are underweight
    "children_with_diarrhoea_pct": 20.0,
    "vitamin_a_supplement_pct": 56.0,
    "full_immunization_pct": 54.0,
    "birth_registration_pct": 32.0,  # % of children under 5 with birth certificate
    "ece_attendance_pct": 13.6,     # Early childhood education attendance (age 3-5)
    "primary_net_attendance_male": 85.0,
    "primary_net_attendance_female": 87.0,
    "primary_net_attendance_total": 86.0,
    "orphan_pct": 8.0,             # % of children <18 who are orphans
}

# UNHS 2019/20 - Household and education context
UNHS_2019_20 = {
    "national_poverty_rate": 20.3,
    "rural_poverty_rate": 24.4,
    "urban_poverty_rate": 11.7,
    "northern_poverty_rate": 32.5,
    "eastern_poverty_rate": 24.1,
    "central_poverty_rate": 10.8,
    "western_poverty_rate": 17.5,
    "avg_household_size": 4.6,
    "rural_household_size": 4.9,
    "urban_household_size": 3.8,
    "primary_ner_total": 79.0,        # Net enrolment ratio (primary)
    "primary_ner_male": 78.0,
    "primary_ner_female": 80.0,
    "primary_completion_rate": 61.0,
    "primary_dropout_rate": 3.3,      # Annual primary dropout rate (%)
    "children_not_in_school_pct": 13.0,
    "main_reason_dropout_cost": 36.0,  # % citing cost
    "main_reason_dropout_distance": 14.0,
    "main_reason_dropout_early_marriage": 8.0,
    "main_reason_dropout_child_labour": 12.0,
    "main_reason_dropout_illness": 9.0,
    "main_reason_dropout_other": 21.0,
}

# UNESCO UIS - Uganda education indicators (selected years)
UNESCO_UIS = {
    "out_of_school_primary_total_2022": 716_000,
    "out_of_school_primary_male_2022": 327_000,
    "out_of_school_primary_female_2022": 389_000,
    "primary_completion_rate_2022": 62.0,
    "gross_enrolment_primary_2022": 104.0,
    "pupil_teacher_ratio_primary_2022": 43.0,
    "govt_expenditure_education_pct_gdp": 2.7,
    "repetition_rate_primary_2022": 10.5,
    "gender_parity_index_primary": 1.02,
    "oos_rate_trend": {  # Out-of-school rate (%) over time
        2015: 16.8, 2016: 15.9, 2017: 14.6, 2018: 13.5,
        2019: 12.7, 2020: 18.3, 2021: 17.1, 2022: 13.0,
    },
    "primary_completion_trend": {
        2015: 54.0, 2016: 56.0, 2017: 57.0, 2018: 59.0,
        2019: 60.0, 2020: 55.0, 2021: 58.0, 2022: 62.0,
    },
}

# Regional dropout risk benchmarks (derived from UNPS + UNHS data)
REGIONAL_BENCHMARKS = {
    "Central": {"dropout_risk": 0.28, "poverty_rate": 0.108, "oos_rate": 0.09},
    "Eastern":  {"dropout_risk": 0.42, "poverty_rate": 0.241, "oos_rate": 0.14},
    "Northern": {"dropout_risk": 0.55, "poverty_rate": 0.325, "oos_rate": 0.19},
    "Western":  {"dropout_risk": 0.35, "poverty_rate": 0.175, "oos_rate": 0.11},
}


def get_interventions(risk_factors):
    """
    Given a dict of risk factors, return a list of recommended interventions
    with priority levels.

    Parameters
    ----------
    risk_factors : dict
        Keys: poor, urban, hsize, welfare_quintile, welfare, region, age, gender

    Returns
    -------
    list of dict with keys: intervention, priority, category, description, why
    """
    interventions = []

    # --- Poverty-triggered ---
    if risk_factors.get("poor", 0) == 1:
        interventions.append({
            "intervention": "School Feeding Programme",
            "priority": "High",
            "category": "Nutrition & Health",
            "description": "Provide daily school meals so children aren't skipping school because of hunger.",
            "why": "36% of dropouts in Uganda cite cost-related reasons. Hungry children struggle to concentrate and attend regularly.",
            "icon": "plate-utensils",
        })
        interventions.append({
            "intervention": "Cash Support (UGX 50,000 per term)",
            "priority": "High",
            "category": "Family Support",
            "description": "Direct cash help to families to cover school fees, uniforms, and supplies.",
            "why": "Families in poverty often cannot afford even 'free' primary education due to hidden costs.",
            "icon": "banknotes",
        })

    # --- Rural ---
    if risk_factors.get("urban", 1) == 0:
        interventions.append({
            "intervention": "Transport Help",
            "priority": "Medium",
            "category": "Getting to School",
            "description": "Bicycle or transport money for children who walk more than 5km to school.",
            "why": "14% of dropouts cite distance to school. Rural children often walk hours each way.",
            "icon": "bicycle",
        })
        interventions.append({
            "intervention": "Mobile Learning Kits",
            "priority": "Medium",
            "category": "Learning Tools",
            "description": "Solar-powered tablets with lessons that work without internet.",
            "why": "Rural schools face teacher shortages (pupil-teacher ratio of 43:1 nationally). Digital tools fill learning gaps.",
            "icon": "tablet",
        })

    # --- Large household ---
    if risk_factors.get("hsize", 0) > LARGE_HOUSEHOLD_THRESHOLD:
        interventions.append({
            "intervention": "Full Scholarship",
            "priority": "High",
            "category": "Family Support",
            "description": "Cover all school costs for children in large families where resources are stretched thin.",
            "why": "Large households (7+ members) spread limited income across many children, increasing dropout risk.",
            "icon": "graduation-cap",
        })
        interventions.append({
            "intervention": "Family Support Services",
            "priority": "Medium",
            "category": "Community Support",
            "description": "Connect families with community health workers and social support services.",
            "why": "Large families benefit from coordinated support including health, nutrition, and education guidance.",
            "icon": "people-group",
        })

    # --- Low welfare (by quintile or absolute value) ---
    low_welfare_q = risk_factors.get("welfare_quintile", 3) <= 2
    low_welfare_abs = (
        risk_factors.get("welfare") is not None
        and float(risk_factors["welfare"]) < LOW_WELFARE_THRESHOLD
    )
    if low_welfare_q or low_welfare_abs:
        interventions.append({
            "intervention": "Free School Supplies",
            "priority": "High",
            "category": "Learning Tools",
            "description": "Books, uniforms, pens, and exercise books provided at no cost to the family.",
            "why": "Families in the bottom 40% of wealth often cannot afford basic school materials.",
            "icon": "book-open",
        })
        interventions.append({
            "intervention": "Parent Skills Training",
            "priority": "Low",
            "category": "Family Empowerment",
            "description": "Help parents and guardians learn new skills to increase household income over time.",
            "why": "Sustainable poverty reduction helps keep children in school long-term.",
            "icon": "chalkboard-user",
        })

    # --- Gender-specific ---
    if risk_factors.get("gender", "").lower() == "female":
        interventions.append({
            "intervention": "Girls' Mentorship Programme",
            "priority": "Medium",
            "category": "Community Support",
            "description": "Pair at-risk girls with female role models who encourage them to stay in school.",
            "why": "Girls face unique barriers including early marriage (8% of dropouts) and household chores.",
            "icon": "user-group",
        })
        interventions.append({
            "intervention": "Hygiene & Sanitary Supplies",
            "priority": "High",
            "category": "Health & Wellbeing",
            "description": "Monthly supply of sanitary products so girls don't miss school during their periods.",
            "why": "Many girls miss up to a week of school each month due to lack of sanitary products.",
            "icon": "heart-pulse",
        })

    # --- Age-specific ---
    if risk_factors.get("age", 0) >= 8:
        interventions.append({
            "intervention": "Child Labour Prevention",
            "priority": "High",
            "category": "Child Protection",
            "description": "Community awareness programmes and monitoring to prevent children from being pulled into work.",
            "why": "12% of dropouts cite child labour. Older children are more likely to be pulled into farm or domestic work.",
            "icon": "shield-halved",
        })

    # Fallback
    if not interventions:
        interventions.append({
            "intervention": "Regular Check-ins",
            "priority": "Low",
            "category": "Monitoring",
            "description": "Teachers and community workers check in regularly to track attendance and wellbeing.",
            "why": "Even low-risk children benefit from consistent monitoring to catch early warning signs.",
            "icon": "clipboard-check",
        })

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    interventions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return interventions


def get_risk_level(probability):
    """Convert a dropout probability to a plain-English risk level and colour."""
    if probability >= 0.7:
        return "High", "#DC3545"
    elif probability >= 0.4:
        return "Medium", "#FD7E14"
    else:
        return "Low", "#198754"


def get_risk_explanation(probability):
    """Return a plain-English explanation of the risk level."""
    if probability >= 0.7:
        return (
            "This child is at **high risk** of dropping out of school. "
            "Immediate action is recommended to address the underlying factors."
        )
    elif probability >= 0.4:
        return (
            "This child faces a **moderate risk** of dropping out. "
            "Preventive measures can significantly improve their chances of staying in school."
        )
    else:
        return (
            "This child is currently at **lower risk** of dropping out, "
            "but continued monitoring is still important."
        )


def compute_heuristic_risk(poor=0, urban=1, hsize=4, welfare=100000.0):
    """Heuristic risk score in [0, 1] based on known risk factors."""
    score = 0.0
    score += 0.35 * float(poor)
    score += 0.20 * (1.0 - float(urban))
    score += 0.20 * min(float(hsize), 20.0) / 20.0
    w = min(max(float(welfare), 0.0), 250_000.0)
    score += 0.25 * (1.0 - w / 250_000.0)
    return round(score, 4)
