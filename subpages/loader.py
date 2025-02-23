#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/23 13:35
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   loader.py
# @Desc     :   

from pandas import DataFrame
from streamlit import empty, sidebar, data_editor

from utilis.faiss import faiss_index_loader, faiss_index_search
from utilis.tools import params_loader

empty_messages: empty = empty()

query: list[str] = ["为什么良好的睡眠对健康至关重要？"]
compares: list[str] = [
    "良好的睡眠有助于身体修复自身，增强免疫系统",
    "在监督学习中，算法经常需要大量的标记数据来进行有效学习",
    "睡眠不足可能导致长期健康问题，如心脏病和糖尿病",
    "这种学习方法依赖于数据质量和数量",
    "它帮助维持正常的新陈代谢和体重控制",
    "睡眠对儿童和青少年的大脑发育和成长尤为重要",
    "良好的睡眠有助于提高日间的工作效率和注意力",
    "监督学习的成功取决于特征选择和算法的选择",
    "量子计算机的发展仍处于早期阶段，面临技术和物理挑战",
    "量子计算机与传统计算机不同，后者使用二进制位进行计算",
    "机器学习使我睡不着觉",
]
sentences: list[str] = query + compares

features: dict[str, list[float]] = {}
differences: dict[str, float] = {}

file_name: str = params_loader()

if file_name != "Select":
    if sidebar.button("Load to Check Similarities", type="primary", help="Load the Faiss index"):
        index = faiss_index_loader(file_name)
        empty_messages.success(f"The Faiss index has been successfully loaded from {file_name}.faiss.")

        for i in range(index.ntotal):
            for sentence in sentences:
                features[sentence] = index.reconstruct(i)

        df_features = DataFrame(features).T
        data_editor(df_features, disabled=True, use_container_width=True)

        distances, indices = faiss_index_search(index, df_features, 3)
        for compare, distance in zip(compares, distances[0]):
            differences[compare] = float(distance)

        df_differences = DataFrame(differences.items(), columns=["Sentence", "Difference"])
        data_editor(df_differences, disabled=True, hide_index=True, use_container_width=True)

        empty_messages.success("The similarities have been successfully calculated.")
else:
    empty_messages.info("Please select a file to load.")
