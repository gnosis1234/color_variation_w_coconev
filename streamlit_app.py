import pdb 

import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit import session_state as ss

from tools.post_process import extract_colorset, color_change, hex_to_rgb, rgb_to_hex

st.set_page_config(
   page_title="Ex-stream-ly Cool App",
   page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)

# Specify canvas parameters in application
with st.sidebar:
    stroke_width = 3
    point_display_radius = 3
    ss.selected_color = st.color_picker("Select Color: ")
    bg_color = "#eee"
    bg_image = st.file_uploader("Background image:", type=["png", "jpg"])

    realtime_update = st.checkbox("Update in realtime", True)

def undo_image():
    ss.index = ss.index - 1 if ss.index > 0 else ss.index
def redo_image():
    ss.index = ss.index + 1 if len(ss.img_queue) > ss.index + 1 else ss.index
def reset_image():
    ss.index = 0
    ss.img_queue = [ss.img_queue[0]]
def select_sub_img(i, j):
    img = ss.img_queue[-1]
    h,w,_=np.shape(img)
    height = h // 9
    width = w // 4
    ss.sub_img = img[i*height:(i + 1) * height, j * width: (j + 1) * width, :]
    ss.sub_img = cv2.resize(ss.sub_img, dsize=(350, 350), interpolation=cv2.INTER_NEAREST)
    ss.sub_h, ss.sub_w, _ = np.shape(ss.sub_img)
def change_color():
    for color, changed_color in zip(ss.palette, ss.changed_palette):

        if not (color == hex_to_rgb(changed_color)).all():
            img = color_change(image=ss.img_queue[ss.index], prev_bgr=color, target_rgb=hex_to_rgb(changed_color))
            ss.img_queue.append(img)
            ss.index += 1
            select_sub_img(ss.i, ss.j)
if bg_image and 'img_queue' not in ss:
    img = Image.open(bg_image).convert("RGB")
    w, h = img.size
    img = np.asarray(img, dtype=np.uint8)
    
    ss.img = img
    ss.img_queue = [img]
    ss.index = 0
    ss.objects = None
    
if bg_image:
    col1, col2 = st.columns(spec=2)
    with col1:
        ss.i = st.number_input("height index", value=0, step=1)
    with col2:
        ss.j = st.number_input("width index", value=0, step=1)
    st.button(label="select the sub image index", on_click=select_sub_img, args=(ss.i, ss.j,))
    
    
col1, col2, col3 = st.columns(spec=3, gap='medium')

# Create a canvas component
if 'sub_img' in ss:
    with col1:
        st.header("Target Image")
        st.image(ss.img)
    with col2:
        st.header("Canvas")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=ss.selected_color,
            background_color=bg_color,
            background_image=Image.fromarray(ss.sub_img) if bg_image else None,
            update_streamlit=realtime_update,
            height=ss.sub_h,
            width=ss.sub_w,
            drawing_mode='point',
            point_display_radius=1,
            key="canvas",
            display_toolbar=False,
        )

        col1_, col2_, col3_, col4_ = st.columns(4)

        with col1_:
            st.button(label="undo", on_click=undo_image)
        with col2_:
            st.button(label="redo", on_click=redo_image)
        with col3_:
            st.button(label="reset", on_click=reset_image)
        with col4_:
            buf = BytesIO()
            Image.fromarray(ss.img_queue[ss.index]).save(buf, format="JPEG")
            byte_im = buf.getvalue()
            st.download_button(label="save", data=buf, file_name=bg_image.name,mime="image/png")
        
    # Do something interesting with the image data and paths

    with col3: 
        st.header("Result Image")
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

            if (not objects.empty) and (ss.objects != objects.to_dict()):
                x = int(objects.at[objects.index[-1],'left'] + objects.at[objects.index[-1],'width'] // 2)
                y = int(objects.at[objects.index[-1],'top'] + objects.at[objects.index[-1],'height'] // 2)
                
                # prev_index = ss.index - 1 if ss.index > 0 else 0
                img = color_change(image=ss.img_queue[ss.index], prev_bgr=ss.sub_img[y][x], target_rgb=hex_to_rgb(ss.selected_color))
                ss.img_queue.append(img)
                ss.index += 1
                select_sub_img(ss.i, ss.j)
            ss.objects = objects.to_dict()

        st.image(ss.img_queue[ss.index])


    st.header("Palette")

    ss.palette = extract_colorset(image=ss.img_queue[ss.index])
    palette_size = len(ss.palette)
    
    columns = st.columns(palette_size)
    ss.changed_palette = []
    for i, col in enumerate(columns):
        with col:        
            color_set = st.color_picker(label=str(i), 
                                        value=rgb_to_hex(ss.palette[i]),
                                        key=f"pal_{i}")
            st.write(f"HEX {rgb_to_hex(ss.palette[i])}")
            st.write(f"RGB {(ss.palette[i])}")
            ss.changed_palette.append(color_set)

    st.button(label="change color", on_click=change_color)
    
    

    