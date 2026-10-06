"""Generate reproducible SUMO training and held-out validation scenarios.

The existing fixed route files remain the evaluation benchmark.  This script
samples only from their already SUMO-valid routes, then changes (1) the number
of vehicles entering from each boundary direction and (2) their departure
times. Training and validation use disjoint seeds. Each scenario writes one
vehicle route file and one pedestrian route file, plus a manifest describing
the exact generated demand.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


DURATION_SECONDS = 3600.0
BASE_VEHICLES = 1800
BASE_PEDESTRIANS = 720
XSI_NAMESPACE = "http://www.w3.org/2001/XMLSchema-instance"

# Percentages are deliberately symmetric within an axis.  A future scenario
# family can add one-sided entrance bias without changing this generator.
PROFILES = {
    "balanced": {"left": 0.25, "right": 0.25, "top": 0.25, "bottom": 0.25},
    "ew_heavy": {"left": 0.35, "right": 0.35, "top": 0.15, "bottom": 0.15},
    "ns_heavy": {"left": 0.15, "right": 0.15, "top": 0.35, "bottom": 0.35},
    "ew_very_heavy": {"left": 0.40, "right": 0.40, "top": 0.10, "bottom": 0.10},
    "ns_very_heavy": {"left": 0.10, "right": 0.10, "top": 0.40, "bottom": 0.40},
}
DEMAND_LEVELS = {
    "low": 0.80,
    "standard": 1.00,
    "high": 1.20,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate direction-balanced SUMO MAPPO training scenarios"
    )
    parser.add_argument("--vehicle-template", default="grid_3x3.rou.xml")
    parser.add_argument("--pedestrian-template", default="grid_3x3_ped_random.rou.xml")
    parser.add_argument("--output-dir", default="training_scenarios")
    parser.add_argument("--duration", type=float, default=DURATION_SECONDS)
    parser.add_argument(
        "--seeds",
        default="101,202",
        help="Comma-separated scenario seeds; two seeds produce 30 training scenarios.",
    )
    parser.add_argument(
        "--validation-seeds",
        default="303",
        help="Comma-separated held-out seeds; seed 303 produces 15 validation scenarios.",
    )
    return parser.parse_args()


def parse_seeds(value, option_name):
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{option_name} must contain comma-separated integers") from error
    if not seeds:
        raise ValueError(f"{option_name} must contain at least one integer")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{option_name} contains duplicate seeds")
    return seeds


def read_templates(path, tag):
    root = ET.parse(path).getroot()
    templates = []
    for element in root.findall(tag):
        templates.append(ET.fromstring(ET.tostring(element, encoding="utf-8")))
    if not templates:
        raise ValueError(f"No <{tag}> templates found in {path}")
    return templates


def entry_direction(vehicle):
    first_edge = vehicle.find("route").attrib["edges"].split()[0]
    for direction in ("left", "right", "top", "bottom"):
        if first_edge.startswith(direction):
            return direction
    raise ValueError(f"Could not classify entry edge {first_edge!r}")


def allocate(total, proportions):
    """Allocate an integer total by largest remainder without changing its sum."""
    raw = {key: total * value for key, value in proportions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda key: (raw[key] - counts[key], key), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def stratified_departures(count, duration, rng):
    """Random arrivals with even coverage over the whole demand interval."""
    departures = [duration * (index + rng.random()) / count for index in range(count)]
    return sorted(departures)


def routes_root():
    ET.register_namespace("xsi", XSI_NAMESPACE)
    return ET.Element(
        "routes",
        {f"{{{XSI_NAMESPACE}}}noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"},
    )


def write_vehicle_routes(path, templates_by_direction, counts, duration, seed):
    rng = random.Random(seed)
    selected = []
    for direction, count in counts.items():
        selected.extend(rng.choices(templates_by_direction[direction], k=count))
    rng.shuffle(selected)

    root = routes_root()
    for index, (template, depart) in enumerate(
        zip(selected, stratified_departures(len(selected), duration, rng))
    ):
        vehicle = ET.fromstring(ET.tostring(template, encoding="utf-8"))
        vehicle.attrib["id"] = f"train_vehicle_{index:05d}"
        vehicle.attrib["depart"] = f"{depart:.2f}"
        root.append(vehicle)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_pedestrian_routes(path, templates, count, duration, seed):
    rng = random.Random(seed)
    selected = rng.choices(templates, k=count)
    rng.shuffle(selected)

    root = routes_root()
    root.append(ET.Element("vType", {"id": "ped_pedestrian", "vClass": "pedestrian"}))
    for index, (template, depart) in enumerate(
        zip(selected, stratified_departures(len(selected), duration, rng))
    ):
        person = ET.fromstring(ET.tostring(template, encoding="utf-8"))
        person.attrib["id"] = f"train_pedestrian_{index:05d}"
        person.attrib["type"] = "ped_pedestrian"
        person.attrib["depart"] = f"{depart:.2f}"
        root.append(person)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    args = parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    train_seeds = parse_seeds(args.seeds, "--seeds")
    validation_seeds = parse_seeds(args.validation_seeds, "--validation-seeds")
    overlap = set(train_seeds) & set(validation_seeds)
    if overlap:
        raise ValueError(
            "training and validation seeds must be disjoint; "
            f"overlap={sorted(overlap)}"
        )

    vehicle_templates = read_templates(Path(args.vehicle_template), "vehicle")
    pedestrian_templates = read_templates(Path(args.pedestrian_template), "person")
    templates_by_direction = defaultdict(list)
    for vehicle in vehicle_templates:
        templates_by_direction[entry_direction(vehicle)].append(vehicle)
    missing = set(PROFILES["balanced"]) - set(templates_by_direction)
    if missing:
        raise ValueError(f"Vehicle templates missing entry directions: {sorted(missing)}")

    output_dir = Path(args.output_dir)
    split_seeds = {
        "train": train_seeds,
        "validation": validation_seeds,
    }
    for split in split_seeds:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "scenario_id",
        "split",
        "profile",
        "demand_level",
        "seed",
        "duration_s",
        "vehicle_total",
        "pedestrian_total",
        "left_vehicles",
        "right_vehicles",
        "top_vehicles",
        "bottom_vehicles",
        "vehicle_route_file",
        "pedestrian_route_file",
    ]

    rows = []
    for split, seeds in split_seeds.items():
        split_dir = output_dir / split
        for profile_name, proportions in PROFILES.items():
            for level_name, scale in DEMAND_LEVELS.items():
                vehicle_total = round(
                    BASE_VEHICLES * scale * args.duration / DURATION_SECONDS
                )
                pedestrian_total = round(
                    BASE_PEDESTRIANS * scale * args.duration / DURATION_SECONDS
                )
                counts = allocate(vehicle_total, proportions)
                for seed in seeds:
                    scenario_id = f"{profile_name}_{level_name}_seed{seed}"
                    vehicle_path = split_dir / f"{scenario_id}_vehicle.rou.xml"
                    pedestrian_path = (
                        split_dir / f"{scenario_id}_pedestrian.rou.xml"
                    )
                    write_vehicle_routes(
                        vehicle_path,
                        templates_by_direction,
                        counts,
                        args.duration,
                        seed,
                    )
                    write_pedestrian_routes(
                        pedestrian_path,
                        pedestrian_templates,
                        pedestrian_total,
                        args.duration,
                        seed + 10_000,
                    )
                    rows.append({
                        "scenario_id": scenario_id,
                        "split": split,
                        "profile": profile_name,
                        "demand_level": level_name,
                        "seed": seed,
                        "duration_s": f"{args.duration:.0f}",
                        "vehicle_total": vehicle_total,
                        "pedestrian_total": pedestrian_total,
                        "left_vehicles": counts["left"],
                        "right_vehicles": counts["right"],
                        "top_vehicles": counts["top"],
                        "bottom_vehicles": counts["bottom"],
                        "vehicle_route_file": str(
                            vehicle_path.relative_to(output_dir)
                        ),
                        "pedestrian_route_file": str(
                            pedestrian_path.relative_to(output_dir)
                        ),
                    })

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    train_count = sum(row["split"] == "train" for row in rows)
    validation_count = sum(row["split"] == "validation" for row in rows)
    print(
        f"Generated {train_count} training and {validation_count} validation "
        f"scenarios in {output_dir}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
