"""
Author: LTH
Date: 2023-05-30 20:00:21
LastEditTime: 2023-05-31 09:44:34
FilePath: \files\python_work\课程\artificial_inteligence\rules.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved. 

"""
import random
import numpy as np
import math

# 双方球门的位置
self_door = [-1, 0]
goal_door = [1, 0]

id_to_act_with_ball = {
    "LF": [8, 12, 5],
    "RF": [2, 12, 5],
    "CF": [5, 12, 13],
    "LM": [5, 6, 13],
    "RM": [5, 4, 13],
    "CM": [5, 13, 15],
    "DM": [5, 17, 13],
    "LD": [6, 17, 13],
    "RD": [4, 17, 13],
    "CD": [5, 17, 5],
}


def softmax(x) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / (2 * np.sum(exp_x))


def make_decision_with_ball(obs_dict: dict) -> int:
    # sourcery skip: avoid-builtin-shadow, merge-else-if-into-elif
    position_x, position_y = obs_dict["left_pos"]
    ops_x, ops_y = obs_dict["right_pos"]
    if -1 * 1.5 / 3.5 <= position_y <= 1 * 1.5 / 3.5:  # C/D
        if position_x >= 0.4:
            id = "CF"
        elif -0.2 <= position_x < 0.4:
            id = "CM"
        elif -0.6 <= position_x < -0.2:
            id = "DM"
        else:
            id = "CD"
    elif position_y > 1 * 1.5 / 3.5:  # L
        if position_x >= 0.4:
            id = "LF"
        elif -0.4 <= position_x < 0.4:
            id = "LM"
        else:
            id = "LD"
    else:  # R
        if position_x >= 0.4:
            id = "RF"
        elif -0.2 <= position_x < 0.4:
            id = "RM"
        else:
            id = "RD"
    possible_action_list = id_to_act_with_ball[id]
    probabilities = [0.75, 0.15, 0.1]
    # 按照给定的的概率执行三个动作
    act = random.choices(possible_action_list, probabilities)[0]
    if calculate_distance(position_x, position_y, ops_x, ops_y) <= 0.15:
        possible_action_list = [
            act,
            change_action(position_x, position_y, ops_x, ops_y),
        ]
        probabilities = [0.4, 0.6]
        act = random.choices(possible_action_list, probabilities)[0]
    return act


def make_decision_without_ball(
    observation_dict: dict,
) -> int:
    act_table = [[2, 3, 4], [1, 0, 5], [8, 7, 6]]
    acts_table_ = [
        [1, 2, 8],
        [2, 1, 3],
        [3, 2, 4],
        [4, 5, 3],
        [5, 4, 6],
        [6, 5, 7],
        [7, 6, 8],
        [8, 1, 7],
    ]
    self_position = np.asarray(observation_dict["left_pos"])
    ops_next_position = np.asarray(observation_dict["right_pos"]) + np.asarray(
        observation_dict["right_velo"]
    )
    # ball_next_position = (
    #     np.asarray(observation_dict["ball_pos"])
    #     + np.asarray(observation_dict["ball_velo"]) / 60
    # )

    ball_next_position = np.asarray(observation_dict["ball_pos"])

    ball_next_position = np.delete(ball_next_position, -1)
    self_ball_delta = ball_next_position - self_position
    p = softmax(np.abs(self_ball_delta)).tolist()

    temp = (np.sign(self_ball_delta) + np.asarray([1, 1])).astype(int).tolist()
    act = act_table[temp[0]][temp[1]]
    possible_action_list = acts_table_[act - 1]  # 第二个是x方向的,第三个是y方向的
    probabilities = ([0.5] + p) if act in [2, 4, 6, 8] else [0.7, 0.15, 0.15]
    return random.choices(possible_action_list, probabilities)[0]


def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def calculate_angle(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 360
    return angle


def change_action(a_x, a_y, b_x, b_y):
    angle = calculate_angle(a_x, a_y, b_x, b_y)
    if angle > 20 or angle < -20:
        return 12
    elif b_y > a_y:
        return 7
    return 3
    # return 12
