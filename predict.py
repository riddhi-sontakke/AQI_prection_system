"""
CLI prediction for the local-file AQI model.

No mock data and no API data is used. The selected city is resolved to a local
CSV in data/, then the city-specific trained artifacts are loaded from models/.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from data.city_data import available_cities
from models.artifact_loader import load_pipeline_for_city


def predict_single(city: str, data_path: str = None) -> dict:
    pipeline = load_pipeline_for_city(city=city, data_path=data_path)
    result = pipeline.predict_now(verbose=True)
    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Predict AQI for a city using only local CSV data and trained artifacts."
    )
    parser.add_argument("--city", default="Bengaluru", choices=available_cities())
    parser.add_argument("--data", default=None, help="Optional local CSV path for this city")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--interval", type=int, default=15)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.schedule:
        pipeline = load_pipeline_for_city(city=args.city, data_path=args.data)
        pipeline.run_scheduler(
            interval_minutes=args.interval,
            output_file=args.json_out or f"logs/{args.city.lower()}_predictions.json",
        )
        return

    result = predict_single(args.city, args.data)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Saved to {args.json_out}")


if __name__ == "__main__":
    main()
