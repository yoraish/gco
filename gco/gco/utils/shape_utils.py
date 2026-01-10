# General imports.
import torch
import random

# Project imports.
from gco.config import Config as cfg

def generate_high_res_shape(shape_type=None):
    """Generate a high-resolution binary shape and its contact points."""
    if shape_type is None:
        shape_type = random.choice(['rectangle', 'triangle', 'half_moon', 't_shape'])
    
    # Initialize high-res mask
    mask = np.zeros((cfg.OBS_PIX, cfg.OBS_PIX), dtype=np.uint8)
    
    if shape_type == 'rectangle':
        w, h = random.randint(48, 128), random.randint(48, 128)
        x0, y0 = (cfg.OBS_PIX-w)//2, (cfg.OBS_PIX-h)//2
        mask[y0:y0+h, x0:x0+w] = 1
        contacts = [(y0+h//2, x0), (y0+h//2, x0+w-1), (y0+h-1, x0+w//2)]
        
    elif shape_type == 'triangle':
        s = random.randint(80, 160)
        x0, y0 = (cfg.OBS_PIX-s)//2, (cfg.OBS_PIX-s)//2
        ori = random.choice(["ul","ur","ll","lr"])
        for i in range(s):
            if   ori=="ul": mask[y0+i,     x0       :x0+s-i] = 1
            elif ori=="ur": mask[y0+i,     x0+i     :x0+s  ] = 1
            elif ori=="ll": mask[y0+s-1-i, x0       :x0+s-i] = 1
            else:           mask[y0+s-1-i, x0+i     :x0+s  ] = 1
        if   ori=="ul": contacts = [(y0,x0+s//2), (y0+s//2,x0), (y0+s//2,x0+s//2)]
        elif ori=="ur": contacts = [(y0,x0+s//2), (y0+s//2,x0+s-1), (y0+s//2,x0+s//2)]
        elif ori=="ll": contacts = [(y0+s-1,x0+s//2), (y0+s//2,x0), (y0+s//2,x0+s//2)]
        else:           contacts = [(y0+s-1,x0+s//2), (y0+s//2,x0+s-1), (y0+s//2,x0+s//2)]
        
    elif shape_type == 'half_moon':
        R = random.randint(48, 96)
        cy = cx = cfg.OBS_PIX // 2
        yy, xx = np.meshgrid(np.arange(cfg.OBS_PIX), np.arange(cfg.OBS_PIX), indexing="ij")
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= R ** 2
        mask[inside & (yy >= cy)] = 1
        contacts = [(cy + R, cx - R//2), (cy + R, cx + R//2), (cy, cx)]
        
    else:  # t_shape
        bar_w = random.randint(96, 160)
        bar_t = random.randint(24, 40)
        stem_w = random.randint(24, 40)
        stem_h = random.randint(64, 112)
        cy = cx = cfg.OBS_PIX // 2
        y_bar = cy - (bar_t + stem_h) // 2
        x_bar = cx - bar_w // 2
        mask[y_bar:y_bar + bar_t, x_bar:x_bar + bar_w] = 1
        x_stem = cx - stem_w // 2
        y_stem = y_bar + bar_t
        mask[y_stem:y_stem + stem_h, x_stem:x_stem + stem_w] = 1
        contacts = [(y_bar + bar_t, x_bar + bar_w//4),
                   (y_bar + bar_t, x_bar + bar_w - 1 - bar_w//4),
                   (y_bar - 1, cx)]
    
    return mask, contacts
