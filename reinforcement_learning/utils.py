import numpy as np

def get_cos_sim(x, y):
    # 将列表转换为 numpy 数组
    x_array = np.array(x)
    y_array = np.array(y)

    # 计算 x 和 y 的点积
    dot_product = np.dot(x_array, y_array)

    # 计算 x 和 y 的向量范数（长度）
    x_norm = np.linalg.norm(x_array)
    y_norm = np.linalg.norm(y_array)
    res = dot_product / (x_norm * y_norm + 1e-6)
    if np.isnan(res):
        res = 0.0

    return res

def get_dis(x, y):
    # 将列表转换为 numpy 数组
    x_array = np.array(x)
    y_array = np.array(y)

    # 计算 x 和 y 之间的差值
    diff = x_array - y_array

    # 计算差值的平方和
    squared_diff = np.square(diff)

    # 计算平方和的平方根（欧几里得距离）
    euclidean_distance = np.sqrt(np.sum(squared_diff))
    
    if np.isnan(euclidean_distance):
        euclidean_distance = 0.0

    return euclidean_distance


def calculating_state(observation):
    observation_dict = {}
    observation_list = []
    pos2 = 115
    pos3 = 115*2
    pos4 = 115*3
    
    avg_pos_left_x = (observation[0]+observation[0+pos2]+observation[0+pos3]+observation[0+pos4])/4
    # avg_pos_left_x = observation[0]
    avg_pos_left_y = (observation[1]+observation[1+pos2]+observation[1+pos3]+observation[1+pos4])/4
    avg_dir_left_x = (observation[22]+observation[22+pos2]+observation[22+pos3]+observation[22+pos4])/4
    avg_dir_left_y = (observation[23]+observation[23+pos2]+observation[23+pos3]+observation[23+pos4])/4
    
    avg_pos_right_x = (observation[44]+observation[44+pos2]+observation[44+pos3]+observation[44+pos4])/4
    avg_pos_right_y = (observation[45]+observation[45+pos2]+observation[45+pos3]+observation[45+pos4])/4
    avg_dir_right_x = (observation[66]+observation[66+pos2]+observation[66+pos3]+observation[66+pos4])/4
    avg_dir_right_y = (observation[67]+observation[67+pos2]+observation[67+pos3]+observation[67+pos4])/4
    
    avg_ball_pos_x = (observation[88]+observation[88+pos2]+observation[88+pos3]+observation[88+pos4])/4
    avg_ball_pos_y = (observation[89]+observation[89+pos2]+observation[89+pos3]+observation[89+pos4])/4
    avg_ball_pos_z = (observation[90]+observation[90+pos2]+observation[90+pos3]+observation[90+pos4])/4
    
    avg_ball_dir_x = (observation[91]+observation[91+pos2]+observation[91+pos3]+observation[91+pos4])/4
    avg_ball_dir_y = (observation[92]+observation[92+pos2]+observation[92+pos3]+observation[92+pos4])/4
    avg_ball_dir_z = (observation[93]+observation[93+pos2]+observation[93+pos3]+observation[93+pos4])/4
    
    control_none = observation[94+pos4]
    control_left = observation[95+pos4]
    control_right = observation[96+pos4]
    
    velo_left_x = observation[0+pos4] - observation[0]
    velo_left_y = observation[1+pos4] - observation[1]
    angv_left_x = observation[22+pos4] - observation[22]
    angv_left_y = observation[23+pos4] - observation[23]
    
    velo_right_x = observation[44+pos4] - observation[44]
    velo_right_y = observation[45+pos4] - observation[45]
    angv_right_x = observation[66+pos4] - observation[66]
    angv_right_y = observation[67+pos4] - observation[67]
    
    velo_ball_x = observation[88+pos4] - observation[88]
    velo_ball_y = observation[89+pos4] - observation[89]
    velo_ball_z = observation[90+pos4] - observation[90]
    
    angv_ball_x = observation[91+pos4] - observation[91]
    angv_ball_y = observation[92+pos4] - observation[92]
    angv_ball_z = observation[93+pos4] - observation[93]

    left_active = observation[97+pos4]
    right_active = observation[108+pos4]
    
    left_right_dis = get_dis([avg_pos_left_x,avg_pos_left_y],[avg_dir_right_x,avg_dir_right_y])
    left_ball_dis = get_dis([avg_pos_left_x,avg_pos_left_y],[avg_ball_pos_x,avg_ball_pos_y])
    right_ball_dis = get_dis([avg_pos_right_x,avg_pos_right_y],[avg_ball_pos_x,avg_ball_pos_y])
    left_ball_dir_ang = get_cos_sim([avg_dir_left_x,avg_dir_left_y],[avg_ball_dir_x,avg_ball_dir_y])
    right_ball_dir_ang = get_cos_sim([avg_dir_right_x,avg_dir_right_y],[avg_ball_dir_x,avg_ball_dir_y])
    left_right_dir_ang = get_cos_sim([avg_dir_left_x,avg_dir_left_y],[avg_dir_right_x,avg_dir_right_y])
    left_to_left_door = get_dis([avg_pos_left_x,avg_pos_left_y],[-1,0])
    left_to_right_door = get_dis([avg_pos_left_x,avg_pos_left_y],[1,0])
    right_to_left_door = get_dis([avg_pos_right_x,avg_pos_right_y],[-1,0])
    right_to_right_door = get_dis([avg_pos_right_x,avg_pos_right_y],[1,0])
    
    left_ball_x = avg_pos_left_x - avg_ball_pos_x
    left_ball_y = avg_pos_left_y - avg_ball_pos_y
    left_right_x = avg_pos_left_x - avg_dir_right_x
    left_right_y = avg_pos_left_y - avg_dir_right_y
    right_ball_x = avg_pos_right_x - avg_ball_pos_x 
    right_ball_y = avg_pos_right_y - avg_ball_pos_y 
    
    left_ball_dis_inv = 1.0 / left_ball_dis
    left_right_dis_inv = 1.0 / left_right_dis
    right_ball_dis_inv = 1.0 / right_ball_dis
    left_to_left_door_inv = 1.0 / left_to_left_door
    left_to_right_door_inv = 1.0 / left_to_right_door
    right_to_left_door_inv = 1.0 / right_to_left_door
    right_to_right_door_inv = 1.0 / right_to_right_door
    #10维其他特征
    
    observation_dict['left_pos'] = [avg_pos_left_x,avg_pos_left_y] #位置
    observation_dict['left_dir'] = [avg_dir_left_x,avg_dir_left_y] #方向
    observation_dict['left_velo'] = [velo_left_x,velo_left_y] #速度
    observation_dict['left_angv'] = [angv_left_x,angv_left_y] #角速
    observation_dict['right_pos'] = [avg_pos_right_x,avg_pos_right_y]
    observation_dict['right_dir'] = [avg_dir_right_x,avg_dir_right_y]
    observation_dict['right_velo'] = [velo_right_x,velo_right_y]
    observation_dict['right_angv'] = [angv_right_x,angv_right_y]
    observation_dict['ball_pos'] = [avg_ball_pos_x,avg_ball_pos_y,avg_ball_pos_z]
    observation_dict['ball_dir'] = [avg_ball_dir_x,avg_ball_dir_y,avg_ball_dir_z]
    observation_dict['ball_velo'] = [velo_ball_x,velo_ball_y,velo_ball_z]
    observation_dict['ball_angv'] = [angv_ball_x,angv_ball_y,angv_ball_z]
    observation_dict['control'] = [control_none,control_left,control_right] #控球
    observation_dict['active'] = [left_active,right_active] # 球员激活
    
    # 还要做的：增加相对位置、增加距离特征；增加相对方向、增加夹角
    
    
    observation_list = [avg_pos_left_x,avg_pos_left_y,avg_dir_left_x,avg_dir_left_y,velo_left_x,velo_left_y,angv_left_x,angv_left_y,avg_pos_right_x,avg_pos_right_y,avg_dir_right_x,avg_dir_right_y,velo_right_x,velo_right_y,angv_right_x,angv_right_y,avg_ball_pos_x,avg_ball_pos_y,avg_ball_pos_z,avg_ball_dir_x,avg_ball_dir_y,avg_ball_dir_z,velo_ball_x,velo_ball_y,velo_ball_z,angv_ball_x,angv_ball_y,angv_ball_z,control_none,control_left,control_right,left_active,right_active,left_right_dir_ang,left_ball_dir_ang,right_ball_dir_ang,left_ball_x,left_ball_y,right_ball_x,right_ball_y,left_right_x,left_right_y,left_right_dis,left_ball_dis,right_ball_dis,left_to_left_door,left_to_right_door,right_to_left_door,right_to_right_door,left_ball_dis_inv,right_ball_dis_inv,left_right_dis_inv,left_to_left_door_inv,left_to_right_door_inv,right_to_left_door_inv,right_to_right_door_inv]
    
    # observation_list = [avg_pos_left_x,avg_pos_left_y,avg_dir_left_x,avg_dir_left_y,avg_pos_right_x,avg_pos_right_y,avg_dir_right_x,avg_dir_right_y,avg_ball_pos_x,avg_ball_pos_y,avg_ball_pos_z,avg_ball_dir_x,avg_ball_dir_y,avg_ball_dir_z,control_none,control_left,control_right]
    
    return observation_dict,observation_list

def calculating_reward(dic_,reward):
    # dic , _ = calculating_state(observation)
    reward_res = 0 #建立奖励累加器
    reward_goal = 0
    if reward < 0 :
        reward_goal = reward * 150
    if reward > 0 :
        reward_goal = reward * 150
    reward_dis =  -(np.linalg.norm(np.array(dic_['left_pos']) - np.array(dic_['ball_pos'][0:2])))* 0.1 #离开球太远，会被惩罚
    # reward_con = dic_['control'][1]*0.2 - dic_['control'][2]*0.4 #控球会有奖励，但是不控球就受到惩罚
    # left_ball_line = np.array(dic['ball_pos'][0:2]) - np.array(dic['left_pos']) #计算从左侧球员指向球的方向向量
    # dot = np.dot(np.array(dic['left_dir']),left_ball_line)
    # cos = dot/(np.linalg.norm(left_ball_line)*np.linalg.norm(np.array(dic['left_dir'])))
    # if np.isnan(cos):
    #     cos = 0
    # reward_dir = cos * 0.10 # 朝着球跑会获得正的奖励
    # reward_atk = dic['ball_pos'][0]*0.1 #球在对方半场加分！ x坐标大于0时在对方半场
    # reward_ball_dir = dic['ball_velo'][0] * 0.1 #球向着对方半场走就加分
    
    reward_atk = (dic_['ball_pos'][0]) * 0.02
    reward_near = 0
    if dic_['ball_pos'][0] > 0.50  and  dic_['control'][2] < 1 and abs(dic_['ball_pos'][1]) < 0.80:
        reward_near += dic_['ball_pos'][0]* 0.04
        # print("level1")
    
    if dic_['ball_pos'][0] > 0.60 and  dic_['control'][2] < 1 and abs(dic_['ball_pos'][1]) < 0.70:
        reward_near += dic_['ball_pos'][0]* 0.06
        # print("level2")
        
    if dic_['ball_pos'][0] > 0.70 and  dic_['control'][2] < 1 and abs(dic_['ball_pos'][1]) < 0.60:
        reward_near += dic_['ball_pos'][0]* 0.12
        # print("level3")
    
    if dic_['ball_pos'][0] > 0.80 and  dic_['control'][2] < 1 and abs(dic_['ball_pos'][1]) < 0.60:
        reward_near += dic_['ball_pos'][0]* 0.20
        # print("level4")
    
    if dic_['ball_pos'][0] < -0.50:
        reward_near -= (np.linalg.norm(np.array(dic_['left_pos']) - np.array(dic_['right_pos'])))* 0.2
    reward_res =  reward_goal + reward_near*0.50 + reward_dis*1 + reward_atk*1
    
    return reward_res  , 0
    

def calculating_MA_states(observation):
    
    # print(observation[0] == observation[1])
    observation = observation[1]
    # print(observation)
    observation_dict = {}    #储存上帝视角可以看到的信息
    observation_dict["left_pos"] = [(observation[0],observation[1]),(observation[2],observation[3]),(observation[4],observation[5])] #存储左侧三个智能体的位置
    observation_dict["right_pos"] = [(observation[0+44],observation[1+44]),(observation[2+44],observation[3+44]),(observation[4+44],observation[5+44])] #存储右侧三个智能体的位置
    observation_dict["ball_pos"] = [(observation[88],observation[89],observation[90])] #球的位置
    observation_dict["left_dir"] = [(observation[0+22],observation[1+22]),(observation[2+22],observation[3+22]),(observation[4+22],observation[5+22])] #存储左侧三个智能体的方向
    
    observation_dict["right_dir"] = [(observation[0+66],observation[1+66]),(observation[2+66],observation[3+66]),(observation[4+66],observation[5+66])] #存储右侧三个智能体的方向
   
    observation_dict["ball_dir"] = [(observation[91],observation[92],observation[93])] #球的方向
    observation_dict["control"] = [(observation[94],observation[95],observation[96])] #控球者

    observation_to_critic = [] #喂给critic的数据，装有全局的所有数据，critic具有上帝视角

    for book in observation_dict:
        for item in observation_dict[book]:
            observation_to_critic.extend(iter(item))
    # observation_dict["left_players_pos"] = left_players_pos
    for lefti in observation_dict["left_pos"]:
        for righti in observation_dict["right_pos"]:
            observation_to_critic.append(get_dis(lefti,righti)) #存入两边队员之间的距离信息
    
    for i in range(3):
        for j in range(i,3):
            observation_to_critic.append(get_dis(observation_dict["left_pos"][i],observation_dict["left_pos"][j])) #存入左侧队员之间的距离信息
    
    for i in range(3):
        for j in range(i,3):
            observation_to_critic.append(get_dis(observation_dict["right_pos"][i],observation_dict["right_pos"][j])) #存入右侧队员之间的距离信息
    
    # 下面需要给出每个智能体能够观测到的状态，这些状态通常都是相对的。这主要包括本队伍其他球员、对手和自己的相对坐标，球与自身的相对坐标，自身与他人运动方向的夹角、控球球员
    observation_to_agent1 = [] 
    # 前四个维度是自己与其他队友的相对位置坐标
    observation_to_agent1.append(observation_dict["left_pos"][1][0] - observation_dict["left_pos"][0][0]) #x坐标之差
    observation_to_agent1.append(observation_dict["left_pos"][2][0] - observation_dict["left_pos"][0][0]) 
    observation_to_agent1.append(observation_dict["left_pos"][1][1] - observation_dict["left_pos"][0][1]) #y坐标之差
    observation_to_agent1.append(observation_dict["left_pos"][2][1] - observation_dict["left_pos"][0][1]) 
    
    # 再六个维度是自己与对手的相对位置坐标
    observation_to_agent1.append(observation_dict["right_pos"][0][0] - observation_dict["left_pos"][0][0]) #x坐标之差
    observation_to_agent1.append(observation_dict["right_pos"][1][0] - observation_dict["left_pos"][0][0]) 
    observation_to_agent1.append(observation_dict["right_pos"][2][0] - observation_dict["left_pos"][0][0])
    observation_to_agent1.append(observation_dict["right_pos"][0][1] - observation_dict["left_pos"][0][1]) #y坐标之差
    observation_to_agent1.append(observation_dict["right_pos"][1][1] - observation_dict["left_pos"][0][1]) 
    observation_to_agent1.append(observation_dict["right_pos"][2][1] - observation_dict["left_pos"][0][1]) 
    
    # 再两个维度是自己与球的相对位置坐标（只考虑x,y坐标）
    observation_to_agent1.append(observation_dict["ball_pos"][0][0] - observation_dict["left_pos"][0][0]) #x坐标之差
    observation_to_agent1.append(observation_dict["ball_pos"][0][1] - observation_dict["left_pos"][0][1]) #y坐标之差
    
    # 再两个维度是队友的运动方向和自己的运动方向的余弦相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["left_dir"][1],observation_dict["left_dir"][0])) #自己和2号队友的相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["left_dir"][2],observation_dict["left_dir"][0])) #自己和3号队友的相似度
    
    # 再三个维度是自己和对面的余弦相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["right_dir"][0],observation_dict["left_dir"][0])) #自己和1号对手的相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["right_dir"][1],observation_dict["left_dir"][0])) #自己和2号对手的相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["right_dir"][2],observation_dict["left_dir"][0])) #自己和3号对手的相似度
    
    
    
    # 再一个维度是自己和球的相似度
    observation_to_agent1.append(get_cos_sim(observation_dict["ball_dir"][0][0:2],observation_dict["left_dir"][0]))
    
    # 再三个维度是谁在控球
    observation_to_agent1.append(observation_dict["control"][0][0])
    observation_to_agent1.append(observation_dict["control"][0][1])
    observation_to_agent1.append(observation_dict["control"][0][2])
    
    observation_to_agent1.append(get_dis(observation_dict["left_pos"][1],observation_dict["left_pos"][0])) #距离信息
    observation_to_agent1.append(get_dis(observation_dict["left_pos"][2],observation_dict["left_pos"][0]))
    observation_to_agent1.append(get_dis(observation_dict["right_pos"][0],observation_dict["left_pos"][0]))
    observation_to_agent1.append(get_dis(observation_dict["right_pos"][1],observation_dict["left_pos"][0]))
    observation_to_agent1.append(get_dis(observation_dict["right_pos"][2],observation_dict["left_pos"][0]))
    
    
    #每个智能体的信息：4+6+2+2+3+1+3+5 = 26 维
    
    # 智能体2看的东西
    observation_to_agent2 = [] 
    # 前四个维度是自己与其他队友的相对位置坐标
    observation_to_agent2.append(observation_dict["left_pos"][0][0] - observation_dict["left_pos"][1][0]) #x坐标之差
    observation_to_agent2.append(observation_dict["left_pos"][2][0] - observation_dict["left_pos"][1][0]) 
    observation_to_agent2.append(observation_dict["left_pos"][0][1] - observation_dict["left_pos"][1][1]) #y坐标之差
    observation_to_agent2.append(observation_dict["left_pos"][2][1] - observation_dict["left_pos"][1][1]) 
    
    # 再六个维度是自己与对手的相对位置坐标
    observation_to_agent2.append(observation_dict["right_pos"][0][0] - observation_dict["left_pos"][1][0]) #x坐标之差
    observation_to_agent2.append(observation_dict["right_pos"][1][0] - observation_dict["left_pos"][1][0]) 
    observation_to_agent2.append(observation_dict["right_pos"][2][0] - observation_dict["left_pos"][1][0])
    observation_to_agent2.append(observation_dict["right_pos"][0][1] - observation_dict["left_pos"][1][1]) #y坐标之差
    observation_to_agent2.append(observation_dict["right_pos"][1][1] - observation_dict["left_pos"][1][1]) 
    observation_to_agent2.append(observation_dict["right_pos"][2][1] - observation_dict["left_pos"][1][1]) 
    
    # 再两个维度是自己与球的相对位置坐标（只考虑x,y坐标）
    observation_to_agent2.append(observation_dict["ball_pos"][0][0] - observation_dict["left_pos"][1][0]) #x坐标之差
    observation_to_agent2.append(observation_dict["ball_pos"][0][1] - observation_dict["left_pos"][1][1]) #y坐标之差
    
    # 再两个维度是队友的运动方向和自己的运动方向的余弦相似度
    observation_to_agent2.append(get_cos_sim(observation_dict["left_dir"][0],observation_dict["left_dir"][1])) #自己和2号队友的相似度
    observation_to_agent2.append(get_cos_sim(observation_dict["left_dir"][2],observation_dict["left_dir"][1])) #自己和3号队友的相似度
    
    # 再三个维度是自己和对面的余弦相似度
    observation_to_agent2.append(get_cos_sim(observation_dict["right_dir"][0],observation_dict["left_dir"][1])) #自己和1号对手的相似度
    observation_to_agent2.append(get_cos_sim(observation_dict["right_dir"][1],observation_dict["left_dir"][1])) #自己和2号对手的相似z度
    observation_to_agent2.append(get_cos_sim(observation_dict["right_dir"][2],observation_dict["left_dir"][1])) #自己和3号对手的相似度
    
    # 再一个维度是自己和球的相似度
    observation_to_agent2.append(get_cos_sim(observation_dict["ball_dir"][0][0:2],observation_dict["left_dir"][1]))
    
    # 再三个维度是谁在控球
    observation_to_agent2.append(observation_dict["control"][0][0])
    observation_to_agent2.append(observation_dict["control"][0][1])
    observation_to_agent2.append(observation_dict["control"][0][2])
    
    observation_to_agent2.append(get_dis(observation_dict["left_pos"][0],observation_dict["left_pos"][1])) #距离信息
    observation_to_agent2.append(get_dis(observation_dict["left_pos"][2],observation_dict["left_pos"][1]))
    observation_to_agent2.append(get_dis(observation_dict["right_pos"][0],observation_dict["left_pos"][1]))
    observation_to_agent2.append(get_dis(observation_dict["right_pos"][1],observation_dict["left_pos"][1]))
    observation_to_agent2.append(get_dis(observation_dict["right_pos"][2],observation_dict["left_pos"][1]))
    
    
    # 同理，写下智能体3所看到的东西
    observation_to_agent3 = [] 
    # 前四个维度是自己与其他队友的相对位置坐标
    observation_to_agent3.append(observation_dict["left_pos"][0][0] - observation_dict["left_pos"][2][0]) #x坐标之差
    observation_to_agent3.append(observation_dict["left_pos"][1][0] - observation_dict["left_pos"][2][0]) 
    observation_to_agent3.append(observation_dict["left_pos"][0][1] - observation_dict["left_pos"][2][1]) #y坐标之差
    observation_to_agent3.append(observation_dict["left_pos"][1][1] - observation_dict["left_pos"][2][1]) 
    
    # 再六个维度是自己与对手的相对位置坐标
    observation_to_agent3.append(observation_dict["right_pos"][0][0] - observation_dict["left_pos"][2][0]) #x坐标之差
    observation_to_agent3.append(observation_dict["right_pos"][1][0] - observation_dict["left_pos"][2][0]) 
    observation_to_agent3.append(observation_dict["right_pos"][2][0] - observation_dict["left_pos"][2][0])
    observation_to_agent3.append(observation_dict["right_pos"][0][1] - observation_dict["left_pos"][2][1]) #y坐标之差
    observation_to_agent3.append(observation_dict["right_pos"][1][1] - observation_dict["left_pos"][2][1]) 
    observation_to_agent3.append(observation_dict["right_pos"][2][1] - observation_dict["left_pos"][2][1]) 
    
    # 再两个维度是自己与球的相对位置坐标（只考虑x,y坐标）
    observation_to_agent3.append(observation_dict["ball_pos"][0][0] - observation_dict["left_pos"][2][0]) #x坐标之差
    observation_to_agent3.append(observation_dict["ball_pos"][0][1] - observation_dict["left_pos"][2][1]) #y坐标之差
    
    # 再两个维度是队友的运动方向和自己的运动方向的余弦相似度
    observation_to_agent3.append(get_cos_sim(observation_dict["left_dir"][0],observation_dict["left_dir"][2])) #自己和2号队友的相似度
    observation_to_agent3.append(get_cos_sim(observation_dict["left_dir"][1],observation_dict["left_dir"][2])) #自己和3号队友的相似度
    
    # 再三个维度是自己和对面的余弦相似度
    observation_to_agent3.append(get_cos_sim(observation_dict["right_dir"][0],observation_dict["left_dir"][2])) #自己和1号对手的相似度
    observation_to_agent3.append(get_cos_sim(observation_dict["right_dir"][1],observation_dict["left_dir"][2])) #自己和2号对手的相似z度
    observation_to_agent3.append(get_cos_sim(observation_dict["right_dir"][2],observation_dict["left_dir"][2])) #自己和3号对手的相似度
    
    # 再一个维度是自己和球的相似度
    observation_to_agent3.append(get_cos_sim(observation_dict["ball_dir"][0][0:2],observation_dict["left_dir"][2]))
    
    # 再三个维度是谁在控球
    observation_to_agent3.append(observation_dict["control"][0][0])
    observation_to_agent3.append(observation_dict["control"][0][1])
    observation_to_agent3.append(observation_dict["control"][0][2])
    
    observation_to_agent3.append(get_dis(observation_dict["left_pos"][0],observation_dict["left_pos"][2])) #距离信息
    observation_to_agent3.append(get_dis(observation_dict["left_pos"][1],observation_dict["left_pos"][2]))
    observation_to_agent3.append(get_dis(observation_dict["right_pos"][0],observation_dict["left_pos"][2]))
    observation_to_agent3.append(get_dis(observation_dict["right_pos"][1],observation_dict["left_pos"][2]))
    observation_to_agent3.append(get_dis(observation_dict["right_pos"][2],observation_dict["left_pos"][2]))
    
    # 接下来，critic可以看到这些所有的东西
    # for i in range(19):
    #     observation_to_critic.append(observation_to_agent1[i])
        
    # for i in range(19):
    #     observation_to_critic.append(observation_to_agent2[i])
        
    # for i in range(19):
    #     observation_to_critic.append(observation_to_agent3[i])
        
    # 送入critic网络的特征共有90维，其中由三个Actor上传的特征为19*3 = 57维，提取的全局特征是33维
    
    return observation_dict , observation_to_critic , observation_to_agent1 , observation_to_agent2 , observation_to_agent3
    
            
    

        
        
            