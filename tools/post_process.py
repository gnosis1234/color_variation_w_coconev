import pdb

import numpy as np

def extract_colorset(image):
    h, w, c = image.shape
    result = np.unique(image.reshape((h * w, c)), axis=0)
    return result


def color_change(image, prev_bgr, target_rgb):
    h, w, c = image.shape
    mask = image != prev_bgr
    mask = mask[:,:,0] | mask[:,:,1] | mask[:,:,2]
    r, g, b = target_rgb
    
    target_img = image.copy()
    target_img[mask == False] = (np.ones((h, w, c)) * [r, g, b])[mask == False]
    
    return target_img

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_hex(rgb: np.ndarray) -> str:
    rgb = rgb.reshape(3)
    return '#{:02X}{:02X}{:02X}'.format(*rgb)