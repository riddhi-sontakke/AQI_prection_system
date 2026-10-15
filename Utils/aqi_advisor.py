"""
=============================================================
AQI PREDICTION SYSTEM - AQI CLASSIFIER & PREVENTIVE MEASURES
=============================================================
Module: aqi_advisor.py
Purpose: Convert AQI predictions into actionable health advice
Optimized for Indian conditions (Pune, Delhi, Mumbai)
Follows: India's CPCB AQI categories + WHO guidelines
=============================================================
"""

from dataclasses import dataclass, field
from typing import Optional
import datetime


# ─────────────────────────────────────────────────────────────
# AQI CATEGORIES (India CPCB Standard)
# ─────────────────────────────────────────────────────────────
AQI_CATEGORIES = [
    {"name": "Good",                    "min": 0,   "max": 50,  "color": "#00E400", "emoji": "🟢"},
    {"name": "Satisfactory",            "min": 51,  "max": 100, "color": "#92D050", "emoji": "🟡"},
    {"name": "Moderately Polluted",     "min": 101, "max": 200, "color": "#FFFF00", "emoji": "🟠"},
    {"name": "Poor",                    "min": 201, "max": 300, "color": "#FF7E00", "emoji": "🔴"},
    {"name": "Very Poor",               "min": 301, "max": 400, "color": "#FF0000", "emoji": "🟣"},
    {"name": "Severe / Hazardous",      "min": 401, "max": 500, "color": "#7E0023", "emoji": "⛔"},
]


@dataclass
class AQIResult:
    """Complete AQI prediction result with advice."""
    aqi_value:        float
    aqi_category:     str
    color_hex:        str
    emoji:            str
    confidence:       float           # 0-1, model confidence
    timestamp:        datetime.datetime = field(default_factory=datetime.datetime.now)
    city:             Optional[str]   = None

    # General precautions
    general_advice:   list = field(default_factory=list)

    # Target-specific advice
    children_advice:  list = field(default_factory=list)
    elderly_advice:   list = field(default_factory=list)
    respiratory_advice: list = field(default_factory=list)
    outdoor_workers:  list = field(default_factory=list)

    # Activity guidance
    outdoor_safe:     bool = True
    mask_required:    bool = False
    mask_type:        str  = "No mask needed"
    air_purifier:     bool = False
    emergency:        bool = False

    # City-specific notes
    city_alert:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "aqi":        round(self.aqi_value, 1),
            "category":   self.aqi_category,
            "color":      self.color_hex,
            "emoji":      self.emoji,
            "confidence": round(self.confidence * 100, 1),
            "timestamp":  self.timestamp.strftime("%Y-%m-%d %H:%M"),
            "city":       self.city or "Unknown",
            "advice": {
                "general":            self.general_advice,
                "children":           self.children_advice,
                "elderly":            self.elderly_advice,
                "respiratory":        self.respiratory_advice,
                "outdoor_workers":    self.outdoor_workers,
            },
            "flags": {
                "outdoor_safe":    self.outdoor_safe,
                "mask_required":   self.mask_required,
                "mask_type":       self.mask_type,
                "air_purifier":    self.air_purifier,
                "emergency":       self.emergency,
            },
            "city_alert": self.city_alert,
        }

    def print_report(self):
        """Print a formatted AQI health report."""
        print("\n" + "═" * 60)
        print(f"  {self.emoji}  AQI HEALTH REPORT")
        print("═" * 60)
        print(f"  📍 City      : {self.city or 'Unknown'}")
        print(f"  🕐 Time      : {self.timestamp.strftime('%Y-%m-%d %H:%M')}")
        print(f"  📊 AQI Value : {self.aqi_value:.1f}")
        print(f"  🏷️  Category  : {self.aqi_category}")
        print(f"  🎯 Confidence: {self.confidence*100:.1f}%")
        print("─" * 60)
        print("  🌍 GENERAL ADVICE:")
        for a in self.general_advice:
            print(f"    • {a}")
        if self.children_advice:
            print("\n  👶 FOR CHILDREN:")
            for a in self.children_advice:
                print(f"    • {a}")
        if self.respiratory_advice:
            print("\n  🫁 FOR RESPIRATORY CONDITIONS:")
            for a in self.respiratory_advice:
                print(f"    • {a}")
        if self.outdoor_workers:
            print("\n  👷 FOR OUTDOOR WORKERS:")
            for a in self.outdoor_workers:
                print(f"    • {a}")
        print(f"\n  😷 Mask Required   : {'YES → ' + self.mask_type if self.mask_required else 'No'}")
        print(f"  🏠 Outdoor Safe    : {'Yes' if self.outdoor_safe else 'NO — Stay indoors'}")
        print(f"  🌀 Air Purifier    : {'Recommended' if self.air_purifier else 'Not needed'}")
        if self.city_alert:
            print(f"\n  ⚠️  CITY ALERT: {self.city_alert}")
        if self.emergency:
            print(f"\n  🚨 EMERGENCY: Follow local authority guidelines!")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────
# CITY-SPECIFIC ALERTS (Indian urban conditions)
# ─────────────────────────────────────────────────────────────
CITY_ALERTS = {
    "Delhi": {
        "winter_high": (
            "Delhi winter smog alert! Vehicle pollution + stubble burning from "
            "Punjab/Haryana. Consider Odd-Even vehicle scheme compliance."
        ),
        "diwali": (
            "Diwali period: PM2.5 spikes expected. Avoid firecrackers. "
            "EPCA emergency measures may be in effect."
        ),
    },
    "Mumbai": {
        "construction": (
            "Mumbai: High construction dust today. Coastal winds may help "
            "by evening. Avoid BKC/Andheri construction zones if possible."
        ),
    },
    "Pune": {
        "diwali_laxmipuja": (
            "Pune: Diwali celebration may spike AQI tonight. "
            "Keep windows closed after 8 PM."
        ),
        "summer_dust": (
            "Summer dust storms from Rajasthan may affect western Pune. "
            "PM10 levels elevated — wear N95 outdoors."
        ),
    },
}

# ─────────────────────────────────────────────────────────────
# AQI ADVISOR CLASS
# ─────────────────────────────────────────────────────────────
class AQIAdvisor:
    """
    Converts predicted AQI values into structured health advice.
    Adapts recommendations for:
    - Different population groups (children, elderly, respiratory)
    - Indian city-specific conditions
    - Time of day and season
    """

    def get_category(self, aqi: float) -> dict:
        """Map AQI value to CPCB category."""
        for cat in AQI_CATEGORIES:
            if cat["min"] <= aqi <= cat["max"]:
                return cat
        # > 500 (rare but possible during disasters)
        return {"name": "Severe / Hazardous", "min": 401, "max": 999,
                "color": "#7E0023", "emoji": "⛔"}

    def generate_advice(
        self,
        aqi: float,
        confidence: float = 0.85,
        city: str = None,
        hour: int = None,
        month: int = None,
    ) -> AQIResult:
        """
        Generate complete AQI health advice for a given prediction.

        Parameters:
            aqi        : Predicted AQI value
            confidence : Model confidence (0-1)
            city       : 'Pune', 'Delhi', 'Mumbai', or None
            hour       : Current hour (0-23)
            month      : Current month (1-12)

        Returns:
            AQIResult with full structured advice
        """
        cat = self.get_category(aqi)
        name = cat["name"]

        # ── Advice lookup table ─────────────────────────────
        ADVICE_MAP = {
            "Good": {
                "general": [
                    "Air quality is excellent. Enjoy outdoor activities freely.",
                    "Ideal conditions for exercise, cycling, and outdoor recreation.",
                    "Windows can be kept open for fresh air circulation.",
                ],
                "children": [
                    "Safe for outdoor play and sports activities.",
                    "No restrictions for school outdoor activities.",
                ],
                "elderly": [
                    "Good day for morning/evening walks.",
                    "Breathing exercises outdoors are beneficial today.",
                ],
                "respiratory": [
                    "Safe for most respiratory patients to go outdoors.",
                    "Continue prescribed medications as normal.",
                ],
                "outdoor_workers": [
                    "No special precautions needed.",
                    "Stay hydrated in warm weather.",
                ],
                "outdoor_safe": True, "mask_required": False,
                "mask_type": "No mask needed",
                "air_purifier": False, "emergency": False,
            },
            "Satisfactory": {
                "general": [
                    "Air quality is acceptable for most people.",
                    "Sensitive individuals may experience mild discomfort.",
                    "Outdoor activities are generally safe.",
                    "Light exercise outdoors is fine for healthy adults.",
                ],
                "children": [
                    "Generally safe for outdoor play.",
                    "Watch for signs of eye irritation or coughing.",
                ],
                "elderly": [
                    "Morning/evening walks are fine. Avoid peak traffic hours.",
                    "Light outdoor activity is acceptable.",
                ],
                "respiratory": [
                    "Sensitive individuals: limit prolonged outdoor exertion.",
                    "Carry inhalers if you have asthma.",
                ],
                "outdoor_workers": [
                    "Normal work schedule. Take regular water breaks.",
                    "Light mask (cloth/surgical) optional during dusty work.",
                ],
                "outdoor_safe": True, "mask_required": False,
                "mask_type": "Optional cloth/surgical mask",
                "air_purifier": False, "emergency": False,
            },
            "Moderately Polluted": {
                "general": [
                    "Reduce prolonged outdoor physical exertion.",
                    "Keep windows partially closed during peak traffic hours.",
                    "Check air quality before outdoor activities.",
                    "Increase water intake — hydration helps flush pollutants.",
                ],
                "children": [
                    "Shorten outdoor play sessions (< 30 min at a time).",
                    "Avoid outdoor sports during 9-11 AM and 6-9 PM.",
                    "Rinse eyes and nose after outdoor activities.",
                ],
                "elderly": [
                    "Avoid strenuous outdoor activities.",
                    "Prefer air-conditioned or indoor environments.",
                    "Morning walks: prefer before 7 AM or after 7 PM.",
                ],
                "respiratory": [
                    "Asthma/COPD patients: stay indoors when possible.",
                    "Always carry emergency inhalers.",
                    "Wear N95 mask if outdoor activity is necessary.",
                    "Consult your doctor if symptoms worsen.",
                ],
                "outdoor_workers": [
                    "Wear 3-ply surgical masks at minimum.",
                    "Take breaks indoors every 90 minutes.",
                    "Avoid construction/roadside work without PPE.",
                ],
                "outdoor_safe": True, "mask_required": True,
                "mask_type": "Surgical/3-ply mask recommended",
                "air_purifier": False, "emergency": False,
            },
            "Poor": {
                "general": [
                    "⚠️ Avoid outdoor activities where possible.",
                    "Wear N95/N99 masks outdoors — mandatory.",
                    "Keep all windows and doors closed.",
                    "Run air purifiers in bedrooms and living areas.",
                    "Avoid burning biomass, garbage, or firecrackers.",
                ],
                "children": [
                    "🚫 No outdoor play or sports.",
                    "Keep children indoors at all times.",
                    "Close school windows and use air purifiers in classrooms.",
                    "Watch for symptoms: cough, eye irritation, breathing difficulty.",
                ],
                "elderly": [
                    "🚫 Do NOT go outdoors.",
                    "Keep all medications ready (BP, asthma, cardiac).",
                    "If breathlessness occurs, seek medical help immediately.",
                ],
                "respiratory": [
                    "🚫 Stay completely indoors.",
                    "Increase inhaler dosage only as prescribed by doctor.",
                    "Keep nebulizer ready if prescribed.",
                    "Emergency: Call 104 (Medical Helpline) if acute symptoms.",
                ],
                "outdoor_workers": [
                    "N95 mask is mandatory — no exceptions.",
                    "Work shifts should be reduced to max 4 hours outdoors.",
                    "Demand employer-provided PPE — it's a legal right.",
                    "Construction and demolition work should be paused.",
                ],
                "outdoor_safe": False, "mask_required": True,
                "mask_type": "N95/N99 mask mandatory",
                "air_purifier": True, "emergency": False,
            },
            "Very Poor": {
                "general": [
                    "🚨 HEALTH EMERGENCY — Minimize ALL outdoor exposure.",
                    "Wear N99/P100 respirator masks outdoors.",
                    "Keep all ventilation systems off — recirculate indoor air.",
                    "Run HEPA air purifiers on maximum setting.",
                    "Avoid any form of outdoor exercise.",
                    "Stock indoor food for 24-48 hours if possible.",
                ],
                "children": [
                    "🚫 Schools should consider closure or online classes.",
                    "Absolute indoor confinement.",
                    "Pediatric inhaler should be at hand.",
                    "Watch closely: if child struggles to breathe — ER immediately.",
                ],
                "elderly": [
                    "🚫 Strictly indoors — no exceptions.",
                    "All cardiac/respiratory patients: contact cardiologist/pulmonologist.",
                    "Have emergency contacts and hospital numbers ready.",
                ],
                "respiratory": [
                    "🚨 SEVERE RISK — Even brief outdoor exposure can trigger attacks.",
                    "Oral corticosteroids should be available (as prescribed).",
                    "Consider hospital admission if condition is unstable.",
                ],
                "outdoor_workers": [
                    "🚫 Outdoor work should be halted by employer.",
                    "If work is unavoidable: full respirator (N99/P100) + goggles.",
                    "Maximum 2-hour outdoor shifts with mandatory indoor breaks.",
                    "Report non-compliance to district labor commissioner.",
                ],
                "outdoor_safe": False, "mask_required": True,
                "mask_type": "N99/P100 full respirator mandatory",
                "air_purifier": True, "emergency": False,
            },
            "Severe / Hazardous": {
                "general": [
                    "🚨🚨 EXTREME EMERGENCY — Air is hazardous.",
                    "Stay indoors with windows and doors completely sealed.",
                    "Place wet towels at door gaps to block polluted air.",
                    "HEPA purifiers at highest setting — all rooms.",
                    "Monitor local government emergency broadcasts.",
                    "Avoid ALL physical activity, even indoors.",
                    "Hydrate heavily — drink 3+ litres of water.",
                ],
                "children": [
                    "🚨 LIFE-THREATENING for children — absolute indoor quarantine.",
                    "NO school. NO outdoor activity. NO exceptions.",
                    "Pediatric emergency kits should be ready.",
                    "If child experiences breathing difficulty: Call 108 (Ambulance).",
                ],
                "elderly": [
                    "🚨 LIFE-THREATENING — All elderly should be indoors.",
                    "Alert family members and neighbors.",
                    "Pre-arrange medical support if any chronic conditions present.",
                ],
                "respiratory": [
                    "🚨 LIFE-THREATENING for all respiratory patients.",
                    "Consider preventive hospital admission.",
                    "Emergency: 108 (Ambulance) | 104 (Medical Helpline).",
                ],
                "outdoor_workers": [
                    "🚫 ALL outdoor work must stop — legal mandate in force.",
                    "Contact State Pollution Control Board if employer forces work.",
                    "SPCB helpline: varies by state — check mpcb.gov.in or cpcb.nic.in",
                ],
                "outdoor_safe": False, "mask_required": True,
                "mask_type": "Avoid outdoors entirely — no mask is fully safe at this level",
                "air_purifier": True, "emergency": True,
            },
        }

        adv = ADVICE_MAP.get(name, ADVICE_MAP["Good"])

        # ── City-specific alert logic ──────────────────────
        city_alert = None
        if city and aqi > 150:
            city_alerts = CITY_ALERTS.get(city, {})
            # Winter alert (Nov–Feb)
            if month and month in [11, 12, 1, 2]:
                city_alert = city_alerts.get("winter_high")
            # Diwali period (Oct–Nov)
            elif month and month in [10, 11]:
                city_alert = city_alerts.get("diwali") or \
                             city_alerts.get("diwali_laxmipuja")
            elif city == "Pune" and month in [3, 4, 5]:
                city_alert = city_alerts.get("summer_dust")

        return AQIResult(
            aqi_value=round(aqi, 1),
            aqi_category=name,
            color_hex=cat["color"],
            emoji=cat["emoji"],
            confidence=confidence,
            city=city,
            timestamp=datetime.datetime.now(),
            general_advice=adv["general"],
            children_advice=adv["children"],
            elderly_advice=adv["elderly"],
            respiratory_advice=adv["respiratory"],
            outdoor_workers=adv["outdoor_workers"],
            outdoor_safe=adv["outdoor_safe"],
            mask_required=adv["mask_required"],
            mask_type=adv["mask_type"],
            air_purifier=adv["air_purifier"],
            emergency=adv["emergency"],
            city_alert=city_alert,
        )


# ─────────────────────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    advisor = AQIAdvisor()

    test_cases = [
        (35,  "Pune",   "Good air day"),
        (85,  "Mumbai", "Satisfactory"),
        (145, "Delhi",  "Moderate pollution"),
        (250, "Delhi",  "Poor — winter smog"),
        (380, "Delhi",  "Very poor — Diwali"),
        (450, "Delhi",  "Hazardous — extreme event"),
    ]

    for aqi, city, desc in test_cases:
        print(f"\n{'─'*60}")
        print(f"Test: {desc} | City: {city} | AQI: {aqi}")
        result = advisor.generate_advice(aqi, city=city, month=11, confidence=0.87)
        result.print_report()
