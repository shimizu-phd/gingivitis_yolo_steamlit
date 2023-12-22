import io

import streamlit as st
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random
import os
import tempfile

tag = ['GI=0', 'GI=1', 'GI=2', 'GI=3']
colors = [(0,255,0), (255,0,0), (255,255,0), (0,0,255)]
n_class = len(tag)
img_size = 1280
n_result = 1

st.title('Dogs/Cats Gingivitis Estimator')
st.write('犬猫の歯肉炎を診断します.')
st.write('奥歯を撮影した画像が必要です.')
st.divider()
with st.expander(':red[注意事項]', expanded=True):
    st.write('本プログラムは技術デモであり、診断結果を保証するものではありません.')
    st.write('歯肉炎の疑いがある場合は、アミカペットクリニックにご相談ください.')
st.divider()

@st.cache_resource
def load_model():
    return YOLO('models/best.pt')


model = load_model()

select = st.radio('解析対象を選択してください.', ('画像', 'カメラ'))

if select == '画像':
    uploaded_file = st.file_uploader("ファイルアップロード", type=['png', 'jpg', 'jpeg', 'webp'], accept_multiple_files=True)

    if uploaded_file:
        results_list = []

        with st.spinner('解析中です...'):
            for img in uploaded_file:
                image = Image.open(img)
                image = image.convert("RGB")
                image_show = np.array(image)

                results = model(image, stream=False, imgsz=1280)  # generator of Results objects
                results_list.append(results)

        for result in results_list:
            img = result[0].plot()
            img_r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img_r, use_column_width=True)
            if result[0].boxes.data.numel() == 0:
                st.write(':red[歯茎を検出できません。他の画像を試してください.]')
            elif max(result[0].boxes.cls) == 0:
                st.write(':green[歯茎は正常です.]')
            elif max(result[0].boxes.cls) == 1:
                st.write(':orange[軽度の歯肉炎です.]')
            elif max(result[0].boxes.cls) == 2:
                st.write(':red[中度の歯肉炎です.]')
            elif max(result[0].boxes.cls) == 3:
                st.write(':red[重度の歯肉炎です.]')
            st.divider()



elif select == 'カメラ':
    uploaded_file = st.camera_input("カメラで撮影")

    if uploaded_file:

        with st.spinner('解析中です...'):
            image = Image.open(uploaded_file)
            image = image.convert("RGB")
            image_show = np.array(image)

            results = model(image, stream=False, imgsz=1280)  # generator of Results objects



        img = results[0].plot()
        img_r = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_r, use_column_width=True)
        if results[0].boxes.data.numel() == 0:
            st.write(':red[歯茎を検出できません。他の画像を試してください.]')
        elif max(results[0].boxes.cls) == 0:
            st.write(':green[歯茎は正常です.]')
        elif max(results[0].boxes.cls) == 1:
            st.write(':orange[軽度の歯肉炎です.]')
        elif max(results[0].boxes.cls) == 2:
            st.write(':red[中度の歯肉炎です.]')
        elif max(results[0].boxes.cls) == 3:
            st.write(':red[重度の歯肉炎です.]')
        st.divider()
