#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/2/22 22:32
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   layout.py
# @Desc     :   

from streamlit import Page, navigation


def pages_setter() -> None:
    """ Set the subpages on the sidebar """
    pages: dict = {
        "page": [
            "subpages/home.py",
            "subpages/embed.py",
            "subpages/loader.py",
        ],
        "title": [
            "Home",
            "Embedding",
            "Faiss Loader",
        ],
        "icon": [
            ":material/home:",
            ":material/polyline:",
            ":material/database:",
        ],
    }

    structure: dict = {
        "Introduction": [
            Page(page=pages["page"][0], title=pages["title"][0], icon=pages["icon"][0]),
        ],
        "Hugging Face Model": [
            Page(page=pages["page"][1], title=pages["title"][1], icon=pages["icon"][1]),
            Page(page=pages["page"][2], title=pages["title"][2], icon=pages["icon"][2]),
        ]
    }
    pg = navigation(structure, position="sidebar", expanded=True)
    pg.run()
