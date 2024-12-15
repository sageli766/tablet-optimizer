from osrparse import Replay
from osrparse import ReplayEventOsu
from osrparse.utils import Mod, Key
from osupyparser import OsuFile
from osupyparser.osu.objects import Circle, Slider, Spinner

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

r = Replay.from_path(r'.\replays_lobotomy\auto_BLOODY_RED.osr')

theta = np.radians(5)

r_data = r.replay_data
print(r_data)

new_replay = []

for event in r_data:
    time = event.time_delta
    x = event.__getattribute__('x')
    y = event.__getattribute__('y')
    keys = event.__getattribute__('keys')

    new_x = (x - 256) * np.cos(theta) - (y - 192) * np.sin(theta) + 256
    new_y = (x - 256) * np.sin(theta) + (y - 192) * np.cos(theta) + 192

    if keys == Key.M1:
        keys = Key.K1 | Key.M1
    elif keys == Key.M2:
        keys = Key.K2 | Key.M2

    new_replay.append(ReplayEventOsu(time_delta=time, x=new_x, y=new_y, keys=keys))

r.replay_data = new_replay

r.mods = Mod.NoFail

r.write_path('./replays_lobotomy/auto_BLOODY_RED_5deg.osr')
