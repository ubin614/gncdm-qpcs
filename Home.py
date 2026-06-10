# -*- coding: utf-8 -*-
"""Streamlit 진입점 — 자동으로 소개 페이지로 이동"""
import streamlit as st
from utils.common import PAGE_CONFIG, sidebar_nav

st.set_page_config(**PAGE_CONFIG)
sidebar_nav()
st.switch_page("pages/1_소개.py")
