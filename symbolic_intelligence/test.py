"""
Author: LTH
Date: 2023-06-02 10:33:32
LastEditTime: 2023-06-02 10:40:29
FilePath: \大作业\symbolic_intelligence\test.py
Description: 
Copyright (c) 2023 by LTH, All Rights Reserved.
"""
a = 5
i = 0
while i < a:
    print(i)
    if i == a - 1:
        increase = int(input("请输入增量:"))
        a += increase
    i += 1
