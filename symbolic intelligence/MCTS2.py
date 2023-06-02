import gfootball.env as football_env
import math
import random
import copy


class Node:
    def __init__(self, state, parent, action, depth, depth_limit):
        self.state = state  # 节点状态（环境）
        self.parent = parent  # 父节点
        self.action = action  # 父节点到当前节点的动作
        self.children = {}  # 子节点集合
        self.wins = 0  # 胜利次数
        self.visits = 0  # 访问次数
        self.untried_actions = list(range(state.action_space.n))  # 未尝试的动作集合
        self.depth = depth  # 节点深度
        self.depth_limit = depth_limit

    def is_terminal(self):
        # 判断当前节点是否为终止节点（比赛是否结束）
        return self.state.is_terminal()

    def is_fully_expanded(self):
        # 判断当前节点是否已完全扩展（所有动作都已尝试过）
        return len(self.untried_actions) == 0

    def get_untried_action(self):
        # 从未尝试的动作集合中随机选择一个动作
        return random.choice(self.untried_actions)

    def expand(self, action):
        # 根据选择的动作扩展当前节点，并返回新的子节点
        next_state = football_env.copy_state(self.state)
        next_state.apply_action(action)
        child_node = Node(next_state, self, action, self.depth + 1)
        self.children[action] = child_node
        self.untried_actions.remove(action)
        return child_node

    def get_best_child(self):
        # 根据UCT算法选择最佳的子节点
        exploration_factor = 1.4  # 调整勘探与利用的权衡参数
        best_score = float("-inf")
        best_child = None

        for child in self.children.values():
            score = (
                child.wins / child.visits
                + exploration_factor * (2 * math.log(self.visits) / child.visits) ** 0.5
            )

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def simulate(self):
        # 模拟一场比赛并返回结果
        env = self.state
        while not self.is_terminal():
            action = random.choice(env.get_legal_actions())
            env.apply_action(action)
        return env.get_game_result()

    def backpropagate(self, result):
        # 将模拟结果更新到当前节点及其路径上的所有节点
        self.visits += 1
        self.wins += result

        if self.parent is not None:
            self.parent.backpropagate(result)

    def get_best_action(self):
        # 获取根节点中访问次数最多的动作（即最佳动作）
        best_action = None
        best_visits = -1

        for action, child in self.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        return best_action

    def is_terminal(self):
        return self.depth == self.depth_limit


def get_best_action_series(root_node):
    node = root_node
    best_actions = []
    while len(node.children) > 0:
        node = node.get_best_child()
        best_actions.append(node.action)
    return best_actions


def monte_carlo_tree_search(env, num_iterations, depth_limit):
    # 初始化根节点
    root_node = Node(env, None, None, 0, depth_limit=depth_limit)

    # 进行指定次数的蒙特卡洛树搜索
    for _ in range(num_iterations):
        node = root_node

        # 选择阶段：根据UCT算法选择下一个要扩展的节点，直到达到叶节点
        while not node.is_terminal():
            if not node.is_fully_expanded():
                # 如果当前节点未完全扩展，则选择一个未扩展的动作并扩展
                node = node.expand(node.get_untried_action())
            else:
                # 如果当前节点已完全扩展，则根据UCT算法选择下一个节点
                node = node.get_best_child()

        # 模拟阶段：从当前叶节点开始模拟一场比赛并获得结果
        result = node.simulate()

        # 反向传播阶段：将模拟结果更新到根节点及其路径上的所有节点
        node.backpropagate(result)

    return get_best_action_series(root_node)


# 创建gfootball环境
env = football_env.create_environment(
    env_name="1_vs_1_easy",
    stacked=False,
    representation="simple115",
    rewards="scoring,checkpoints",
    logdir="/tmp/gfootball",
    write_goal_dumps=False,
    write_full_episode_dumps=False,
    render=False,
)

# 执行蒙特卡洛树搜索并获取最佳动作
best_actions = monte_carlo_tree_search(env, num_iterations=1000, depth_limit=10)
print("2个动作内最佳动作序列:", best_actions)
