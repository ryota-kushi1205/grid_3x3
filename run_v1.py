import argparse
import csv
import os
import sys


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

# c0/c2は東西方向へ渡る歩行者、c1/c3は南北方向へ渡る歩行者。
# 同じグループの2横断歩道は同じ現示で同時に青になる。
PEDESTRIAN_GROUPS = {
    "east_west": ("c0", "c2"),
    "north_south": ("c1", "c3"),
}

# 歩行者を流すため現示を切り替えた場合、新しく停止する車両方向。
AFFECTED_VEHICLE_DIRECTIONS = {
    "east_west": ("north", "south"),
    "north_south": ("east", "west"),
}

# 各流入方向の直進信号linkIndex。赤信号で待つ車両の判定に使う。
VEHICLE_THROUGH_LINK_INDEX = {
    "south": 1,
    "east": 5,
    "north": 9,
    "west": 13,
}

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

# SUMOも0.1 m/s以下を停止判定の既定値として使う。
STOP_SPEED = 0.1  # m/s

# 乗用車のアイドル燃料を約0.3 gal/h、ガソリン低位発熱量を
# 約114,000 BTU/galとした代表値（約10 kW）。全車両で固定する。
IDLE_POWER = 10_000.0  # J/(台・s) = W/台

# 1500 kgの乗用車を道路上限13.89 m/sまで再加速する燃料エネルギーの
# 代表値。1/2*m*v^2を変換効率約30%で割った約0.48 MJを丸めた値。
# 実際の発進には加算せず、歩行者現示で新しく止める車両の予測だけに使う。
RESTART_ENERGY = 500_000.0  # J/台

# MUTCDの最小WALK時間、FHWAの歩行者列のWALK時間式、
# 本ネットワークの横断歩道寸法を使う。
# 基本時間 = 最小WALK 7.0 s + 6.4 m / 1.07 m/s = 約13.0 s
# 列放出時間 = 3.2 s + (0.57 / 幅4.0 m) * 人数
PEDESTRIAN_MIN_WALK_TIME = 7.0  # s
PEDESTRIAN_QUEUE_STARTUP_TIME = 3.2  # s
CROSSING_LENGTH = 6.4  # m
PEDESTRIAN_WALK_SPEED = 1.07  # m/s
CROSSWALK_WIDTH = 4.0  # m
T_MARGIN = PEDESTRIAN_MIN_WALK_TIME + (
    CROSSING_LENGTH / PEDESTRIAN_WALK_SPEED
)
T_PER_PED = 0.57 / CROSSWALK_WIDTH  # s/人


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
# 方向別の車両検知
# ============================================================

def get_detector_vehicle_ids_by_direction(junction):
    return {
        direction: set(
            traci.lanearea.getLastStepVehicleIDs(
                f"e2_{junction}_{direction}"
            )
        )
        for direction in DETECTOR_DIRECTIONS
    }


def get_stopped_red_vehicles(junction, vehicles_by_direction):
    """赤信号の流入方向で、現在0.1 m/s未満の車両を返す。"""
    signal_state = traci.trafficlight.getRedYellowGreenState(junction)
    stopped = set()

    for direction, vehicle_ids in vehicles_by_direction.items():
        link_index = VEHICLE_THROUGH_LINK_INDEX[direction]
        if signal_state[link_index].lower() != "r":
            continue

        stopped.update(
            vehicle_id
            for vehicle_id in vehicle_ids
            if traci.vehicle.getSpeed(vehicle_id) < STOP_SPEED
        )

    return stopped


def get_affected_vehicles_by_pedestrian_group(
    crossing_counts,
    vehicles_by_direction,
):
    """
    待機歩行者を流すために現示を切り替えた場合、新しく止める車両を
    方向別E2検知器から近似する。右左折・直進の経路判定は行わない。
    """
    affected = {group: set() for group in PEDESTRIAN_GROUPS}

    for group, crossings in PEDESTRIAN_GROUPS.items():
        if not any(crossing_counts[crossing] > 0 for crossing in crossings):
            continue

        for direction in AFFECTED_VEHICLE_DIRECTIONS[group]:
            affected[group].update(
                vehicle_id
                for vehicle_id in vehicles_by_direction[direction]
                if traci.vehicle.getSpeed(vehicle_id) >= STOP_SPEED
            )

    return affected


# ============================================================
# エネルギー混雑度
# ============================================================

def calculate_pedestrian_prediction(crossing_counts, affected):
    """
    歩行者グループを流す場合の予測負荷を計算する。

    同じ現示で青になる2横断歩道は並行して処理されるため、クリア時間は
    2本の待機人数の最大値で決め、影響車両を1回だけ数える。
    再加速エネルギーは予測だけに含め、実際の発進時には計上しない。
    """
    clear_times = {}
    energy_by_group = {}
    average_power_by_group = {}

    for group, crossings in PEDESTRIAN_GROUPS.items():
        waiting_count = max(
            crossing_counts[crossing]
            for crossing in crossings
        )

        if waiting_count == 0:
            clear_times[group] = 0.0
            energy_by_group[group] = 0.0
            average_power_by_group[group] = 0.0
            continue

        crossing_clearance_time = CROSSING_LENGTH / PEDESTRIAN_WALK_SPEED
        queue_clear_time = (
            PEDESTRIAN_QUEUE_STARTUP_TIME
            + T_PER_PED * waiting_count
            + crossing_clearance_time
        )
        clear_time = max(T_MARGIN, queue_clear_time)
        predicted_energy = len(affected[group]) * (
            IDLE_POWER * clear_time + RESTART_ENERGY
        )

        clear_times[group] = clear_time
        energy_by_group[group] = predicted_energy
        average_power_by_group[group] = predicted_energy / clear_time

    return clear_times, energy_by_group, average_power_by_group


# ============================================================
# 出力
# ============================================================

CSV_FIELDS = [
    "time_s",
    "junction",
    "stopped_red_vehicles",
    "idle_power_mw",
    "wait_c0",
    "wait_c1",
    "wait_c2",
    "wait_c3",
    "wait_east_west",
    "wait_north_south",
    "unresolved_pedestrians",
    "affected_vehicles_for_east_west_ped",
    "affected_vehicles_for_north_south_ped",
    "clear_time_east_west_s",
    "clear_time_north_south_s",
    "predicted_energy_east_west_mj",
    "predicted_energy_north_south_mj",
    "predicted_average_power_mw",
    "total_congestion_mw",
    "reward",
]


def make_output_row(
    time_s,
    junction,
    stopped_red_vehicles,
    crossing_counts,
    unresolved,
    affected,
    clear_times,
    predicted_energy,
    predicted_average_power,
):
    idle_power_mw = len(stopped_red_vehicles) * IDLE_POWER / 1_000_000.0
    prediction_power_mw = (
        sum(predicted_average_power.values()) / 1_000_000.0
    )
    total_congestion_mw = idle_power_mw + prediction_power_mw

    return {
        "time_s": f"{time_s:.0f}",
        "junction": junction,
        "stopped_red_vehicles": len(stopped_red_vehicles),
        "idle_power_mw": f"{idle_power_mw:.6f}",
        "wait_c0": crossing_counts["c0"],
        "wait_c1": crossing_counts["c1"],
        "wait_c2": crossing_counts["c2"],
        "wait_c3": crossing_counts["c3"],
        "wait_east_west": (
            crossing_counts["c0"] + crossing_counts["c2"]
        ),
        "wait_north_south": (
            crossing_counts["c1"] + crossing_counts["c3"]
        ),
        "unresolved_pedestrians": unresolved,
        "affected_vehicles_for_east_west_ped": len(
            affected["east_west"]
        ),
        "affected_vehicles_for_north_south_ped": len(
            affected["north_south"]
        ),
        "clear_time_east_west_s": f"{clear_times['east_west']:.3f}",
        "clear_time_north_south_s": f"{clear_times['north_south']:.3f}",
        "predicted_energy_east_west_mj": (
            f"{predicted_energy['east_west'] / 1_000_000.0:.6f}"
        ),
        "predicted_energy_north_south_mj": (
            f"{predicted_energy['north_south'] / 1_000_000.0:.6f}"
        ),
        "predicted_average_power_mw": f"{prediction_power_mw:.6f}",
        "total_congestion_mw": f"{total_congestion_mw:.6f}",
        "reward": f"{-total_congestion_mw:.6f}",
    }


def print_interval(time_s, rows):
    print()
    print("=" * 126)
    print(
        f"T = {time_s:.0f} s（現在の赤信号アイドル負荷＋"
        "歩行者処理の予測平均負荷）"
    )

    network_idle = 0.0
    network_prediction = 0.0

    for row in rows:
        idle_mw = float(row["idle_power_mw"])
        prediction_mw = float(row["predicted_average_power_mw"])
        total_mw = float(row["total_congestion_mw"])
        network_idle += idle_mw
        network_prediction += prediction_mw

        waits = "/".join(str(row[f"wait_c{i}"]) for i in range(4))
        affected = "/".join(
            str(row[field])
            for field in (
                "affected_vehicles_for_east_west_ped",
                "affected_vehicles_for_north_south_ped",
            )
        )

        print(
            f"{row['junction']} | "
            f"IDLE={idle_mw:.3f} MW "
            f"(STOPPED_RED={row['stopped_red_vehicles']}) | "
            f"PED_PRED={prediction_mw:.3f} MW | "
            f"CONGESTION={total_mw:.3f} MW | "
            f"WAIT(c0/c1/c2/c3)={waits} | "
            f"AFFECTED(EW_PED/NS_PED)={affected} | "
            f"UNRESOLVED={row['unresolved_pedestrians']}"
        )

    print(
        f"NETWORK | IDLE={network_idle:.3f} MW | "
        f"PED_PRED={network_prediction:.3f} MW | "
        f"CONGESTION={network_idle + network_prediction:.3f} MW"
    )


# ============================================================
# メイン処理
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="車両・歩行者の瞬間エネルギー負荷による混雑度v1"
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

    traci.start([
        sumo_binary,
        "-c",
        SUMO_CFG,
        "--step-length",
        str(SIMULATION_STEP),
        "--end",
        str(args.end),
    ])

    try:
        with open(args.output, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while traci.simulation.getTime() < args.end:
                traci.simulationStep()
                time_s = traci.simulation.getTime()

                vehicles_by_junction = {
                    junction: get_detector_vehicle_ids_by_direction(junction)
                    for junction in INTERSECTIONS
                }

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

                    stopped_red_vehicles = get_stopped_red_vehicles(
                        junction,
                        vehicles_by_junction[junction],
                    )
                    affected = get_affected_vehicles_by_pedestrian_group(
                        crossing_counts,
                        vehicles_by_junction[junction],
                    )
                    (
                        clear_times,
                        predicted_energy,
                        predicted_average_power,
                    ) = calculate_pedestrian_prediction(crossing_counts, affected)
                    row = make_output_row(
                        time_s,
                        junction,
                        stopped_red_vehicles,
                        crossing_counts,
                        unresolved,
                        affected,
                        clear_times,
                        predicted_energy,
                        predicted_average_power,
                    )
                    rows.append(row)
                    writer.writerow(row)

                csv_file.flush()
                print_interval(time_s, rows)

    finally:
        traci.close()


if __name__ == "__main__":
    main()
