"""
Author: LTH
Date: 2023-05-28 23:26:15
LastEditTime: 2023-05-28 23:26:43
FilePath: \files\python_work\课程\artificial_inteligence\MCTS.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved. 

"""
import numpy as np
import gfootball as gf
import baselines


class player:
    def __init__(self, player_position: list) -> None:
        """初始化球员位置和速度(0)

        Args:
            player_position (list): 一个含有两个元素的列表，分别为球员开局时的x,y坐标
        """
        self.x, self.y = player_position
        self.x_velocity, self.y_velocity = [0, 0]

    def run(self, position_change: list, velocity_change: list) -> None:
        """_summary_

        Args:
            position_change (list): _description_
            velocity_change (list): _description_
        """
        self.x += position_change[0]
        self.y += position_change[1]
        self.x_velocity += velocity_change[0]
        self.y_velocity += velocity_change[1]


class football:
    def __init__(self, football_position: list, starter: player) -> None:
        """初始化足球位置和速度(0)

        Args:
            football_position (list): 一个含有两个元素的列表，分别为足球开局时的x,y坐标
            starter (player): 开球的球员
        """
        self.x, self.y = football_position
        self.x_velocity, self.y_velocity = [0, 0]
        self.controller = starter

    def kick(
        self, position_change: list, velocity_change: list, kicker: player
    ) -> None:
        self.x += position_change[0]
        self.y += position_change[1]
        self.x_velocity += velocity_change[0]
        self.y_velocity += velocity_change[1]
        self.controller = kicker


class football_game:
    def __init__(
        self,
        player1: player,
        player2: player,
        ball: football,
        goal_position: list,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.ball = ball
        self.goal_position = goal_position


class node:
    def __init__(self, depth: int, depth_limit: int, cp: float) -> None:
        self.average_winning_percentage = 0
        self.wining = 0
        self.lose = 0
        self.depth = depth
        self.depth_limit = depth_limit
        self.children = []
        if depth < depth_limit:
            self.children.extend(node(depth + 1, cp) for _ in range(9))
        self.change_table = [
            [0, 1],
            [1, 1],
            [1, 0],
            [1, -1],
            [0, -1],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, 0],
        ]  # 顺时针一圈的8个方向和保持位置不变，与下标相同的node对应
        self.cp = cp

    def UCT(self) -> int:  # 回答最优的选择（下标）和数值，还原路径的话把第一个元素放一起就行
        wining_rate = [
            (self.wining / (self.wining + self.lose))
            + 2
            * self.cp
            * np.sqrt(2 * np.log(self.wining + self.lose) / (i.wining + i.lose))
            for i in self.children
        ]
        return [np.argmax(wining_rate), np.max(wining_rate)]


class MCTS:
    # 怎么评估胜负？用sigmoid映射后再用一个random来评估胜负？
    # 对手的反应是同时的还是回合制？
    # 如果是同时的，那么对手的反应怎么做
    def __init__(
        self,
        game: football_game,
        evaluate_function: callable,
        depth_limit: int,
        cp: float,
        upper_limit_of_sampling_times: int,
    ) -> None:
        self.game = game
        self.evaluate_function = evaluate_function
        self.root = node(0, depth_limit, cp)
        self.upper_limit_of_sampling_times = upper_limit_of_sampling_times
