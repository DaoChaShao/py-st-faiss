#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/22 22:33
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, divider, expander, caption, empty

title("Vector Manipulation")
divider()
with expander("Application Introduction", expanded=True):
    caption("This application is a simple demonstration of vector manipulation.")
    caption("1. Embedding: Embedding is the process of converting text data into numerical data.")
    caption("2. Faiss: Faiss is a library that helps to perform similarity search.")

empty_message: empty = empty()

empty_message.info("You can navigate to the model page calling the functions.")
