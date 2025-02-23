#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/22 22:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   embed.py
# @Desc     :   

from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from streamlit import (empty, sidebar, spinner, data_editor, plotly_chart,
                       session_state, header, button, columns, markdown)

from utilis.faiss import (faiss_index_creator, faiss_index_adder,
                          faiss_index_search, faiss_index_storager,
                          file_size_getter)
from utilis.tools import (params_model_getter, Timer, DimensionsReducer,
                          params_plotly_getter, sentences_3d)

if "features" not in session_state:
    session_state.features = {}

empty_messages: empty = empty()

model_name: str = params_model_getter()
if model_name != "Select":

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

    if sidebar.button("Embedding", type="primary", help="Embed the sentences"):
        with spinner("Embedding the sentences", show_time=True):
            with Timer("Embedding the sentences") as timer:
                # Load the model
                model = SentenceTransformer(model_name)
                # Embed the sentences
                embeddings = model.encode(sentences)
            empty_messages.success(timer)

            for sentence, embedding in zip(sentences, embeddings):
                session_state.features[sentence] = embedding

    if session_state.features:
        df: DataFrame = DataFrame(session_state.features).T
        data_editor(df, disabled=True, use_container_width=True)

        # Reduce the dimensions
        df_reduce: DataFrame = DimensionsReducer().fit_transform(df, neighbors=5)

        # Get the parameters of the plotly
        point_size, font_size = params_plotly_getter()

        if sidebar.button("3D Chart", type="primary", help="Click to display the 3D chart"):
            # Display the 3D chart
            chart = sentences_3d(df_reduce, point_size, font_size)
            plotly_chart(chart, use_container_width=True)
            empty_messages.success("The sentences have been embedded successfully")

        with sidebar:
            header("Vector Actions")
            col_left, _, col_right = columns([2.5, 0.8, 1.2])

            with col_left:
                button_left = button("Similarities Check", type="secondary",
                                     help="Click to check the similarities between the query and the compares")
            with col_right:
                button_right = button("Save", type="secondary", help="Click to save the Faiss Index")

        if "index" not in session_state:
            session_state.index = None

        if button_left:
            # Create a Faiss index
            session_state.index = faiss_index_creator(df.shape[1])
            # Add the embeddings to the index
            faiss_index_adder(session_state.index, df)
            # Search the index for the similarities
            distances, indices = faiss_index_search(session_state.index, df, 3)

            # Display the similarities
            similarities: dict[str: list[float]] = {}
            for compare, distance in zip(compares, distances[0]):
                similarities[compare] = float(distance)

            empty_messages.success("The similarities have been checked successfully")

            # Display the similarities
            markdown(f"**{query[0]}**")
            data_editor(
                DataFrame(list(similarities.items()), columns=["Sentence", "Similarity"]),
                hide_index=True, disabled=True, use_container_width=True
            )

        if button_right:
            # Save the Faiss index
            faiss_name: str = "medical"
            faiss_index_storager(session_state.index, faiss_name)
            # Get the size of the file
            file_size = file_size_getter(faiss_name)
            empty_messages.success(f"The Faiss Index has been saved successfully, whose size is {file_size}")
else:
    empty_messages.error("Please select a model")
