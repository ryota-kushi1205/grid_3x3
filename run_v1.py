import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET


# ============================================================
# SUMO・TraCIの読み込み
# ============================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("SUMO_HOME が設定されていません")

sys.path.insert(0, os.path.join(os.environ["SUMO_HOME"], "tools"))

import traci


# ============================================================
# 実験条件・混雑度パラメータ（v1）
# ============================================================

SUMO_CFG = "grid_3x3.sumocfg"
NET_FILE = "grid_3x3_ped.net.xml"
CSV_FILE = "congestion_v1.csv"

SIMULATION_END = 3600.0
SIMULATION_STEP = 1.0
DECISION_INTERVAL = 5

INTERSECTIONS = [
    "A0", "A1", "A2",
    "B0", "B1", "B2",
    "C0", "C1", "C2",
]

WALKING_AREAS = ["w0", "w1", "w2", "w3"]
CROSSINGS = ["c0", "c1", "c2", "c3"]
DETECTOR_DIRECTIONS = ["east", "north", "south", "west"]

# 現在のネットワークでは全9交差点でc0～c3が16～19に対応する。
PED_LINK_INDEX = {
    "c0": 16,
    "c1": 17,
    "c2": 18,
    "c3": 19,
}

WALKING_AREA_CROSSINGS = {
    "w0": ["c0", "c3"],
    "w1": ["c0", "c1"],
    "w2": ["c1", "c2"],
    "w3": ["c2", "c3"],
}

STOP_SPEED = 0.1  # m/s
IDLE_POWER = 9_500.0  # J/(台・s)
RESTART_ENERGY = 500_000.0  # J/台
T_MARGIN = 12.0  # s
T_PER_PED = 0.8  # s/人

# SUMOのgetFuelConsumption()はmg/sを返す。
# ガソリンの低位発熱量を暫定44 MJ/kgとして燃料エネルギーへ換算する。
FUEL_ENERGY_PER_MG = 44.0  # J/mg


# ============================================================
# 歩行者待機人数
# ============================================================

def get_stopped_peds(junction, walking_area):
    edge_id = f":{junction}_{walking_area}"
    stopped_count = 0

    for person_id in traci.edge.getLastStepPersonIDs(edge_id):
        if traci.person.getSpeed(person_id) < STOP_SPEED:
            stopped_count += 1

    return stopped_count


def get_ped_signal_states(junction):
    state = traci.trafficlight.getRedYellowGreenState(junction)
    return {
        crossing: state[index]
        for crossing, index in PED_LINK_INDEX.items()
    }


def is_red(signal_state):
    return signal_state.lower() == "r"


def infer_waiting_crossings(w_counts, signals):
    """
    Walking Area内で停止している歩行者を、隣接する赤信号の
    横断歩道へ割り当てる。候補が一意でなければUNRESOLVEDとする。
    """
    crossing_counts = {crossing: 0 for crossing in CROSSINGS}
    unresolved = 0

    for walking_area, stopped_count in w_counts.items():
        if stopped_count == 0:
            continue

        red_crossings = [
            crossing
            for crossing in WALKING_AREA_CROSSINGS[walking_area]
            if is_red(signals[crossing])
        ]

        if len(red_crossings) == 1:
            crossing_counts[red_crossings[0]] += stopped_count
        else:
            unresolved += stopped_count

    return crossing_counts, unresolved


# ============================================================
# 車両進路と横断歩道の競合関係
# ============================================================

def load_crossing_conflict_links(net_file):
    """
    .net.xmlのjunction/request/foesから、各横断歩道と競合する
    車両信号linkIndexを読み取る。

    SUMOのfoesビット列は右端がlinkIndex=0に対応する。
    """
    root = ET.parse(net_file).getroot()
    conflict_links = {}

    for junction in INTERSECTIONS:
        junction_node = root.find(f"./junction[@id='{junction}']")
        if junction_node is None:
            raise RuntimeError(f"{junction}: junction定義がありません")

        requests = {
            int(node.attrib["index"]): node
            for node in junction_node.findall("request")
        }
        conflict_links[junction] = {}

        for crossing, crossing_index in PED_LINK_INDEX.items():
            if crossing_index not in requests:
                raise RuntimeError(
                    f"{junction}/{crossing}: request定義がありません"
                )

            foes = requests[crossing_index].attrib["foes"]
            conflict_links[junction][crossing] = {
                link_index
                for link_index in range(crossing_index)
                if link_index < len(foes)
                and foes[-1 - link_index] == "1"
            }

    return conflict_links


def build_movement_link_maps():
    """
    (流入エッジ, 流出エッジ)から車両信号linkIndexを引く表を作る。
    現在の1車線ネットワークを前提とする。
    """
    movement_maps = {}

    for junction in INTERSECTIONS:
        movement_map = {}
        controlled_links = traci.trafficlight.getControlledLinks(junction)

        for link_index, link_group in enumerate(controlled_links):
            if link_index >= min(PED_LINK_INDEX.values()):
                continue

            for incoming_lane, outgoing_lane, _via_lane in link_group:
                incoming_edge = traci.lane.getEdgeID(incoming_lane)
                outgoing_edge = traci.lane.getEdgeID(outgoing_lane)
                key = (incoming_edge, outgoing_edge)

                if key in movement_map and movement_map[key] != link_index:
                    raise RuntimeError(
                        f"{junction}: 車両進路{key}のlinkIndexが一意ではありません"
                    )

                movement_map[key] = link_index

        movement_maps[junction] = movement_map

    return movement_maps


def get_detector_vehicle_ids(junction):
    vehicle_ids = set()

    for direction in DETECTOR_DIRECTIONS:
        detector_id = f"e2_{junction}_{direction}"
        vehicle_ids.update(
            traci.lanearea.getLastStepVehicleIDs(detector_id)
        )

    return vehicle_ids


def get_vehicle_movement_link(vehicle_id, movement_map):
    route = traci.vehicle.getRoute(vehicle_id)
    route_index = traci.vehicle.getRouteIndex(vehicle_id)

    if route_index < 0 or route_index + 1 >= len(route):
        return None

    current_edge = traci.vehicle.getRoadID(vehicle_id)
    next_edge = route[route_index + 1]
    return movement_map.get((current_edge, next_edge))


def get_affected_vehicles(
    junction,
    crossing_counts,
    vehicle_ids,
    movement_map,
    conflict_links,
):
    """
    待機歩行者がいる横断歩道ごとに、E2内の車両のうち
    その横断歩道と進路が競合する車両IDを抽出する。
    """
    affected = {crossing: set() for crossing in CROSSINGS}

    movement_by_vehicle = {
        vehicle_id: get_vehicle_movement_link(vehicle_id, movement_map)
        for vehicle_id in vehicle_ids
    }

    for crossing in CROSSINGS:
        if crossing_counts[crossing] == 0:
            continue

        foes = conflict_links[junction][crossing]
        affected[crossing] = {
            vehicle_id
            for vehicle_id, link_index in movement_by_vehicle.items()
            if link_index is not None and link_index in foes
        }

    return affected


# ============================================================
# エネルギー混雑度
# ============================================================

def new_vehicle_accumulator():
    return {
        "moving_energy_j": 0.0,
        "idle_energy_j": 0.0,
        "moving_vehicle_seconds": 0.0,
        "stopped_vehicle_seconds": 0.0,
    }


def accumulate_vehicle_energy(accumulator, vehicle_ids):
    """現在の1秒分の移動・停止車両エネルギーを加算する。"""
    for vehicle_id in vehicle_ids:
        speed = traci.vehicle.getSpeed(vehicle_id)

        if speed < STOP_SPEED:
            accumulator["idle_energy_j"] += (
                IDLE_POWER * SIMULATION_STEP
            )
            accumulator["stopped_vehicle_seconds"] += SIMULATION_STEP
        else:
            fuel_mg_per_s = max(
                0.0,
                traci.vehicle.getFuelConsumption(vehicle_id),
            )
            accumulator["moving_energy_j"] += (
                fuel_mg_per_s
                * FUEL_ENERGY_PER_MG
                * SIMULATION_STEP
            )
            accumulator["moving_vehicle_seconds"] += SIMULATION_STEP


def calculate_pedestrian_energy(crossing_counts, affected):
    """
    歩行者を流す場合の予測エネルギーを横断歩道別に計算する。

    現在の1秒は車両側ですでに計上するため、予測アイドル時間から
    SIMULATION_STEPを引く。再発進エネルギーは1台1回と仮定する。
    """
    clear_times = {}
    energy_by_crossing = {}

    for crossing in CROSSINGS:
        waiting_count = crossing_counts[crossing]

        if waiting_count == 0:
            clear_times[crossing] = 0.0
            energy_by_crossing[crossing] = 0.0
            continue

        clear_time = T_MARGIN + T_PER_PED * waiting_count
        future_idle_time = max(0.0, clear_time - SIMULATION_STEP)
        affected_count = len(affected[crossing])

        clear_times[crossing] = clear_time
        energy_by_crossing[crossing] = affected_count * (
            IDLE_POWER * future_idle_time + RESTART_ENERGY
        )

    return clear_times, energy_by_crossing


# ============================================================
# 出力
# ============================================================

CSV_FIELDS = [
    "time_s",
    "junction",
    "moving_vehicle_seconds",
    "stopped_vehicle_seconds",
    "moving_energy_mj",
    "idle_energy_mj",
    "vehicle_congestion_mj",
    "wait_c0",
    "wait_c1",
    "wait_c2",
    "wait_c3",
    "unresolved_pedestrians",
    "affected_c0",
    "affected_c1",
    "affected_c2",
    "affected_c3",
    "clear_time_c0_s",
    "clear_time_c1_s",
    "clear_time_c2_s",
    "clear_time_c3_s",
    "pedestrian_congestion_mj",
    "total_congestion_mj",
]


def make_output_row(
    time_s,
    junction,
    accumulator,
    crossing_counts,
    unresolved,
    affected,
    clear_times,
    ped_energy,
):
    moving_mj = accumulator["moving_energy_j"] / 1_000_000.0
    idle_mj = accumulator["idle_energy_j"] / 1_000_000.0
    vehicle_mj = moving_mj + idle_mj
    pedestrian_mj = sum(ped_energy.values()) / 1_000_000.0

    return {
        "time_s": f"{time_s:.0f}",
        "junction": junction,
        "moving_vehicle_seconds": (
            f"{accumulator['moving_vehicle_seconds']:.1f}"
        ),
        "stopped_vehicle_seconds": (
            f"{accumulator['stopped_vehicle_seconds']:.1f}"
        ),
        "moving_energy_mj": f"{moving_mj:.6f}",
        "idle_energy_mj": f"{idle_mj:.6f}",
        "vehicle_congestion_mj": f"{vehicle_mj:.6f}",
        "wait_c0": crossing_counts["c0"],
        "wait_c1": crossing_counts["c1"],
        "wait_c2": crossing_counts["c2"],
        "wait_c3": crossing_counts["c3"],
        "unresolved_pedestrians": unresolved,
        "affected_c0": len(affected["c0"]),
        "affected_c1": len(affected["c1"]),
        "affected_c2": len(affected["c2"]),
        "affected_c3": len(affected["c3"]),
        "clear_time_c0_s": f"{clear_times['c0']:.1f}",
        "clear_time_c1_s": f"{clear_times['c1']:.1f}",
        "clear_time_c2_s": f"{clear_times['c2']:.1f}",
        "clear_time_c3_s": f"{clear_times['c3']:.1f}",
        "pedestrian_congestion_mj": f"{pedestrian_mj:.6f}",
        "total_congestion_mj": f"{vehicle_mj + pedestrian_mj:.6f}",
    }


def print_interval(time_s, rows):
    print()
    print("=" * 118)
    print(f"T = {time_s:.0f} s（直前{DECISION_INTERVAL}秒の車両実消費＋現在の歩行者予測消費）")

    network_vehicle = 0.0
    network_pedestrian = 0.0

    for row in rows:
        vehicle_mj = float(row["vehicle_congestion_mj"])
        pedestrian_mj = float(row["pedestrian_congestion_mj"])
        total_mj = float(row["total_congestion_mj"])
        network_vehicle += vehicle_mj
        network_pedestrian += pedestrian_mj

        waits = "/".join(str(row[f"wait_c{i}"]) for i in range(4))
        affected = "/".join(
            str(row[f"affected_c{i}"]) for i in range(4)
        )

        print(
            f"{row['junction']} | "
            f"VEH={vehicle_mj:.3f} MJ "
            f"(MOVE={float(row['moving_energy_mj']):.3f}, "
            f"IDLE={float(row['idle_energy_mj']):.3f}) | "
            f"PED={pedestrian_mj:.3f} MJ | "
            f"TOTAL={total_mj:.3f} MJ | "
            f"WAIT(c0/c1/c2/c3)={waits} | "
            f"AFFECTED={affected} | "
            f"UNRESOLVED={row['unresolved_pedestrians']}"
        )

    print(
        f"NETWORK | VEH={network_vehicle:.3f} MJ | "
        f"PED={network_pedestrian:.3f} MJ | "
        f"TOTAL={network_vehicle + network_pedestrian:.3f} MJ"
    )


# ============================================================
# メイン処理
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="車両・歩行者のエネルギー換算混雑度v1"
    )
    parser.add_argument(
        "--nogui",
        action="store_true",
        help="sumo-guiではなくsumoを使用する",
    )
    parser.add_argument(
        "--output",
        default=CSV_FILE,
        help=f"CSV出力先（既定: {CSV_FILE}）",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=SIMULATION_END,
        help=f"終了時刻（既定: {SIMULATION_END:.0f}秒）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sumo_binary = "sumo" if args.nogui else "sumo-gui"
    conflict_links = load_crossing_conflict_links(NET_FILE)

    traci.start([
        sumo_binary,
        "-c",
        SUMO_CFG,
        "--step-length",
        str(SIMULATION_STEP),
        "--end",
        str(args.end),
    ])

    accumulators = {
        junction: new_vehicle_accumulator()
        for junction in INTERSECTIONS
    }

    try:
        movement_maps = build_movement_link_maps()

        with open(args.output, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while traci.simulation.getTime() < args.end:
                traci.simulationStep()
                time_s = traci.simulation.getTime()

                vehicles_by_junction = {
                    junction: get_detector_vehicle_ids(junction)
                    for junction in INTERSECTIONS
                }

                for junction in INTERSECTIONS:
                    accumulate_vehicle_energy(
                        accumulators[junction],
                        vehicles_by_junction[junction],
                    )

                if int(round(time_s)) % DECISION_INTERVAL != 0:
                    continue

                rows = []

                for junction in INTERSECTIONS:
                    w_counts = {
                        walking_area: get_stopped_peds(
                            junction,
                            walking_area,
                        )
                        for walking_area in WALKING_AREAS
                    }
                    signals = get_ped_signal_states(junction)
                    crossing_counts, unresolved = infer_waiting_crossings(
                        w_counts,
                        signals,
                    )

                    stopped_total = sum(w_counts.values())
                    if stopped_total != sum(crossing_counts.values()) + unresolved:
                        raise RuntimeError(
                            f"{junction}: 歩行者数の集計が一致しません"
                        )

                    affected = get_affected_vehicles(
                        junction,
                        crossing_counts,
                        vehicles_by_junction[junction],
                        movement_maps[junction],
                        conflict_links,
                    )
                    clear_times, ped_energy = calculate_pedestrian_energy(
                        crossing_counts,
                        affected,
                    )
                    row = make_output_row(
                        time_s,
                        junction,
                        accumulators[junction],
                        crossing_counts,
                        unresolved,
                        affected,
                        clear_times,
                        ped_energy,
                    )
                    rows.append(row)
                    writer.writerow(row)

                csv_file.flush()
                print_interval(time_s, rows)
                accumulators = {
                    junction: new_vehicle_accumulator()
                    for junction in INTERSECTIONS
                }

    finally:
        traci.close()


if __name__ == "__main__":
    main()
