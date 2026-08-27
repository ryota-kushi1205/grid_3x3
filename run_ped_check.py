import os
import sys


# ============================================================
# SUMO・TraCIの読み込み
# ============================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("SUMO_HOME が設定されていません")

sys.path.insert(
    0,
    os.path.join(os.environ["SUMO_HOME"], "tools")
)

import traci


# ============================================================
# 基本設定
# ============================================================

SUMO_CFG = "grid_3x3.sumocfg"

INTERSECTIONS = [
    "A0", "A1", "A2",
    "B0", "B1", "B2",
    "C0", "C1", "C2"
]

WALKING_AREAS = [
    "w0", "w1", "w2", "w3"
]

CROSSINGS = [
    "c0", "c1", "c2", "c3"
]

# 今回のネットワークでは、
# 歩行者用横断歩道c0～c3がlinkIndex 16～19に対応する
PED_LINK_INDEX = {
    "c0": 16,
    "c1": 17,
    "c2": 18,
    "c3": 19
}

# 0.1m/s未満を停止とみなす
STOP_SPEED = 0.1


# ============================================================
# Walking Areaと横断歩道の位置関係
# ============================================================

# 各交差点の配置
#
#                 c2
#          w3 ----------- w2
#          |               |
#       c3 |               | c1
#          |               |
#          w0 ----------- w1
#                 c0
#
# 各Walking Areaは、2つの横断歩道に隣接する。

WALKING_AREA_CROSSINGS = {
    "w0": ["c0", "c3"],
    "w1": ["c0", "c1"],
    "w2": ["c1", "c2"],
    "w3": ["c2", "c3"]
}


# ============================================================
# Walking Area別の停止歩行者数
# ============================================================

def get_stopped_peds(junction, walking_area):
    """
    指定したWalking Area内で停止している歩行者数を取得する。
    """

    edge_id = f":{junction}_{walking_area}"

    person_ids = traci.edge.getLastStepPersonIDs(edge_id)

    stopped_count = 0

    for person_id in person_ids:

        speed = traci.person.getSpeed(person_id)

        if speed < STOP_SPEED:
            stopped_count += 1

    return stopped_count


# ============================================================
# 歩行者信号状態
# ============================================================

def get_ped_signal_states(junction):
    """
    指定した交差点のc0～c3の歩行者信号状態を取得する。
    """

    state = traci.trafficlight.getRedYellowGreenState(junction)

    return {
        crossing: state[index]
        for crossing, index in PED_LINK_INDEX.items()
    }


def is_red(signal_state):
    """
    信号状態が赤であるかを判定する。
    """

    return signal_state.lower() == "r"


# ============================================================
# 横断歩道別の待機人数推定
# ============================================================

def infer_waiting_crossings(w_counts, signals):
    """
    Walking Area別停止人数と信号状態から、
    c0～c3の横断歩道別待機人数を推定する。

    仮定：
    Walking Area内で速度が0に近い歩行者は、
    隣接する2つの横断歩道のうち、
    赤信号になっている横断歩道を待っている。

    同じ横断歩道の両側にいる待機者は合計する。
    """

    crossing_counts = {
        crossing: 0
        for crossing in CROSSINGS
    }

    unresolved = 0

    for walking_area, stopped_count in w_counts.items():

        if stopped_count == 0:
            continue

        adjacent_crossings = (
            WALKING_AREA_CROSSINGS[walking_area]
        )

        red_crossings = [
            crossing
            for crossing in adjacent_crossings
            if is_red(signals[crossing])
        ]

        # 隣接する2信号のうち、赤が1つだけの場合
        if len(red_crossings) == 1:

            waiting_crossing = red_crossings[0]

            crossing_counts[waiting_crossing] += (
                stopped_count
            )

        # 両方赤、または赤がない場合は判別不能
        else:
            unresolved += stopped_count

    return crossing_counts, unresolved


# ============================================================
# メイン処理
# ============================================================

def main():

    traci.start([
        "sumo-gui",
        "-c",
        SUMO_CFG
    ])

    try:

        while traci.simulation.getMinExpectedNumber() > 0:

            traci.simulationStep()

            t = traci.simulation.getTime()

            print()
            print("=" * 130)
            print(f"T = {t:.0f} s")

            for junction in INTERSECTIONS:

                # --------------------------------------------
                # Walking Area別の停止人数
                # --------------------------------------------

                w_counts = {
                    walking_area: get_stopped_peds(
                        junction,
                        walking_area
                    )
                    for walking_area in WALKING_AREAS
                }

                stopped_total = sum(w_counts.values())

                # --------------------------------------------
                # 歩行者信号状態
                # --------------------------------------------

                signals = get_ped_signal_states(junction)

                # --------------------------------------------
                # 横断歩道別の待機人数を推定
                # --------------------------------------------

                crossing_counts, unresolved = (
                    infer_waiting_crossings(
                        w_counts,
                        signals
                    )
                )

                assigned_total = sum(
                    crossing_counts.values()
                )

                # 集計が一致しているか確認
                if stopped_total != assigned_total + unresolved:
                    raise RuntimeError(
                        f"{junction}: 歩行者数の集計が一致しません"
                    )

                # --------------------------------------------
                # 出力
                # --------------------------------------------

                print(
                    f"{junction} | "
                    f"WAIT_CROSSING "
                    f"c0={crossing_counts['c0']} "
                    f"c1={crossing_counts['c1']} "
                    f"c2={crossing_counts['c2']} "
                    f"c3={crossing_counts['c3']} "
                    f"| STOP_TOTAL={stopped_total} "
                    f"ASSIGNED={assigned_total} "
                    f"UNRESOLVED={unresolved} "
                    f"| SIGNAL "
                    f"c0={signals['c0']} "
                    f"c1={signals['c1']} "
                    f"c2={signals['c2']} "
                    f"c3={signals['c3']} "
                    f"| WALKING_AREA "
                    f"w0={w_counts['w0']} "
                    f"w1={w_counts['w1']} "
                    f"w2={w_counts['w2']} "
                    f"w3={w_counts['w3']}"
                )

    finally:
        traci.close()


if __name__ == "__main__":
    main()