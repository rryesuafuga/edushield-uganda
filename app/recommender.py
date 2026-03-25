"""
EduShield Uganda - Intervention Recommender
Recommends evidence-based interventions based on student risk factors.
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
    list of dict with keys: intervention, priority, category, description
    """
    interventions = []

    # --- Poverty-triggered ---
    if risk_factors.get("poor", 0) == 1:
        interventions.append({
            "intervention": "School Feeding Program",
            "priority": "High",
            "category": "Nutrition",
            "description": "Provide daily school meals to reduce hunger-related absenteeism.",
        })
        interventions.append({
            "intervention": "Cash Transfer (UGX 50,000/term)",
            "priority": "High",
            "category": "Financial",
            "description": "Direct cash transfers to offset schooling costs for poor households.",
        })

    # --- Rural ---
    if risk_factors.get("urban", 1) == 0:
        interventions.append({
            "intervention": "Transport Support",
            "priority": "Medium",
            "category": "Access",
            "description": "Bicycle or transport subsidy for students walking >5km to school.",
        })
        interventions.append({
            "intervention": "Mobile Learning Kits",
            "priority": "Medium",
            "category": "Technology",
            "description": "Solar-powered tablets with offline curriculum content.",
        })

    # --- Large household ---
    if risk_factors.get("hsize", 0) > LARGE_HOUSEHOLD_THRESHOLD:
        interventions.append({
            "intervention": "Targeted Scholarship",
            "priority": "High",
            "category": "Financial",
            "description": "Full tuition scholarship for children in large households.",
        })
        interventions.append({
            "intervention": "Family Planning Education",
            "priority": "Medium",
            "category": "Social",
            "description": "Community health worker visits providing family planning information and referrals.",
        })

    # --- Low welfare (by quintile or absolute value) ---
    low_welfare_q = risk_factors.get("welfare_quintile", 3) <= 2
    low_welfare_abs = (
        risk_factors.get("welfare") is not None
        and float(risk_factors["welfare"]) < LOW_WELFARE_THRESHOLD
    )
    if low_welfare_q or low_welfare_abs:
        interventions.append({
            "intervention": "Free Scholastic Materials",
            "priority": "High",
            "category": "Educational",
            "description": "Books, uniforms, and supplies provided at no cost.",
        })
        interventions.append({
            "intervention": "Economic Empowerment Training",
            "priority": "Low",
            "category": "Livelihood",
            "description": "Skills training for parents/guardians to improve household income.",
        })

    # --- Gender-specific ---
    if risk_factors.get("gender", "").lower() == "female":
        interventions.append({
            "intervention": "Girls' Mentorship Program",
            "priority": "Medium",
            "category": "Social",
            "description": "Pair at-risk girls with female role models in the community.",
        })
        interventions.append({
            "intervention": "Sanitary Supplies Kit",
            "priority": "High",
            "category": "Health",
            "description": "Monthly supply of sanitary products to reduce absenteeism.",
        })

    # --- Age-specific ---
    if risk_factors.get("age", 0) >= 14:
        interventions.append({
            "intervention": "Anti-Child Labour Awareness",
            "priority": "High",
            "category": "Protection",
            "description": "Community campaigns and monitoring against child labour.",
        })

    # Fallback
    if not interventions:
        interventions.append({
            "intervention": "Routine Monitoring",
            "priority": "Low",
            "category": "General",
            "description": "Regular attendance tracking and teacher check-ins.",
        })

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    interventions.sort(key=lambda x: priority_order.get(x["priority"], 3))

    return interventions


def get_risk_level(probability):
    """Convert a dropout probability to a risk level label and color."""
    if probability >= 0.7:
        return "High", "#ff4b4b"
    elif probability >= 0.4:
        return "Medium", "#ffa726"
    else:
        return "Low", "#66bb6a"


def compute_heuristic_risk(poor=0, urban=1, hsize=4, welfare=100000.0):
    """Heuristic risk score in [0, 1] based on known risk factors."""
    score = 0.0
    score += 0.35 * float(poor)
    score += 0.20 * (1.0 - float(urban))
    score += 0.20 * min(float(hsize), 20.0) / 20.0
    w = min(max(float(welfare), 0.0), 250_000.0)
    score += 0.25 * (1.0 - w / 250_000.0)
    return round(score, 4)
