from osrparse import Replay
from osupyparser import OsuFile
from osupyparser.osu.objects import Circle, Slider, Spinner
from osrparse.utils import Mod, Key
from stacking import stacking_fix

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt

# print numpy arrays without scientific notation
np.set_printoptions(suppress=True)



def norm_2(x1, x2, y1, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def dist_from_center(x, y):
    return norm_2(256, x, 192, y)


def solve_theta(x1, x2, y1, y2, degrees=True):
    r_circle = np.array([x1 - 256, y1 - 192])
    r_hit = np.array([x2 - 256, y2 - 192])

    dot_prod = np.dot(r_circle, r_hit)
    cross_prod = r_circle[0] * r_hit[1] - r_circle[1] * r_hit[0]

    angle = np.arccos(dot_prod / (np.linalg.norm(r_circle) * np.linalg.norm(r_hit)))

    if cross_prod < 0:
        angle *= -1

    return angle

# class HitDetector:
#     def __init__(self, map, replay, debug=False, plot=False, human_learning_rate=1.5):
#         self.map = map
#         self.replay = replay
#         self.debug = debug
#         self.plot = plot
#         self.human_learning_rate = 1.5
#         self.hits_circles_array = []

debug = False
plot = True

human_learning_rate = 1.5

# IMPORT REPLAY AND MAP
# r = Replay.from_path(r'.\replays_lobotomy\razorfruit_BLOODY_RED.osr')
# o_map = OsuFile('BLOODY_RED.osu').parse_file()

r = Replay.from_path(r'.\replays_gmtn\razorfruit_gmtn.osr')
o_map = OsuFile('gmtn.osu').parse_file()

r_data = r.replay_data
m_data = o_map.hit_objects

cs, od = o_map.cs, o_map.od

# MAP STATS
radius = 54.4 - 4.48 * cs
window_300 = 80 - 6 * od
window_100 = 140 - 8 * od
window_50 = 200 - 10 * od

hit_data = []
map_data = []

m_data = stacking_fix(m_data, o_map, radius, False)

# CONVERT REPLAY AND MAP DATA INTO NUMPY ARRAYS
for i, event in enumerate(r_data):
    time_delta = event.__getattribute__('time_delta')
    x = event.__getattribute__('x')
    y = event.__getattribute__('y')
    keys = event.__getattribute__('keys')

    hit_data.append([time_delta, x, y, keys])

for i, circle in enumerate(m_data):
    if isinstance(circle, Spinner):
        continue

    x, y = circle.pos.x, circle.pos.y
    time = circle.start_time

    map_data.append([time, x, y])

# convert to numpy arrays
hit_data = np.array(hit_data)
map_data = np.array(map_data)

# change relative time to absolute time
hit_data[:, 0] = np.cumsum(hit_data[:, 0])

# filter hit attempts for only k1, k2 presses
key_presses = hit_data[:, 3]

mask = [False]

for i, curr_event in enumerate(hit_data[1:]):
    prev_hit = hit_data[i][3]
    curr_hit = curr_event[3]

    curr_hit = Key(int(curr_hit))
    prev_hit = Key(int(prev_hit))

    if curr_hit <= 0:
        mask.append(False)
    elif (Key.K1 in prev_hit and Key.K2 in prev_hit) and (Key.K1 in curr_hit and Key.K2 not in curr_hit):
        mask.append(False)
    elif (Key.K1 in prev_hit and Key.K2 in prev_hit) and (Key.K1 not in curr_hit and Key.K2 in curr_hit):
        mask.append(False)
    elif prev_hit == curr_hit:
        mask.append(False)
    elif prev_hit <= 0 and (Key.K1 in curr_hit or Key.K2 in curr_hit):
        mask.append(True)
    elif Key.K1 in prev_hit and Key.K2 in curr_hit:
        mask.append(True)
    elif Key.K2 in prev_hit and Key.K1 in curr_hit:
        mask.append(True)
    else:
        mask.append(False)

hit_attempts = hit_data[mask]

# Mask for debugging
# mask_range = [31456 - 110, 31966 + 110]
# print(hit_data[(hit_data[:, 0] > mask_range[0]) & (hit_data[:, 0] < mask_range[1])])
# print('\n')
# print(hit_attempts[(hit_attempts[:, 0] > mask_range[0]) & (hit_attempts[:, 0] < mask_range[1])])

hit_errors = []

# hit_attempts: [time, x, y, keys]
# map_data: [time, x, y]

start_idx = 0
miss_count = 0
hit_count = 0
assigned = [False] * len(hit_attempts)

# TODO: for circles with no associated hit, search and find the timestamp with the closest time to the circle time
# TODO: round hit errors off

for hit_circle in map_data:
    miss = True
    min_delay = 999999
    min_delay_idx = 0
    i = start_idx

    x_circle = hit_circle[1]
    y_circle = hit_circle[2]

    # Upper and lower 50 timing windows for each hit circle
    timing_upper = hit_circle[0] + window_50
    timing_lower = hit_circle[0] - window_50

    time = hit_attempts[start_idx][0]

    while time < timing_upper and i < len(hit_attempts):
        time = hit_attempts[i][0]
        if hit_attempts[i][0] > timing_lower and not isinstance(assigned[i], np.ndarray):
            # find distance between hit attempt and hit circle
            x_hit = hit_attempts[i][1]
            y_hit = hit_attempts[i][2]
            hit_error = norm_2(x_circle, x_hit, y_circle, y_hit)

            # store the closest attempt for later use
            delay = abs(hit_circle[0] - hit_attempts[i][0])
            if delay < min_delay:
                min_delay_idx = i
                min_delay = delay

            # if the attempted hit is within the circle's radius,
            # record and mark the hit attempt as used and continue
            # and set new starting index as the location of the assigned hit
            if hit_error <= radius:
                hit_errors.append((x_circle - x_hit, y_circle - y_hit))
                if debug:
                    print(f'HIT time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
                assigned[i] = hit_circle
                miss = False
                hit_count += 1
                start_idx = i
                break

        i += 1

    if miss and not assigned[min_delay_idx]:
        x_hit = hit_attempts[min_delay_idx][1]
        y_hit = hit_attempts[min_delay_idx][2]
        hit_errors.append((x_circle - x_hit, y_circle - y_hit))
        if debug:
            print(f'MISS time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
        assigned[min_delay_idx] = hit_circle
        start_idx = min_delay_idx
        miss_count += 1

if debug:
    print(f'Miss count: {miss_count}, Hit count: {hit_count}')

hit_error_x = np.array([x for x, y in hit_errors])
hit_error_y = np.array([y for x, y in hit_errors])

d_theta = []
d_size = []

for i, circle in enumerate(assigned):
    if isinstance(circle, np.ndarray):
        hit_attempt = hit_attempts[i]

        x_hit = hit_attempt[1]
        y_hit = hit_attempt[2]

        x_circ = circle[1]
        y_circ = circle[2]

        dx = x_circ - x_hit
        dy = y_circ - y_hit

        r_circ = dist_from_center(x_circ, y_circ)
        r_hit = dist_from_center(x_hit, y_hit)
        r_error = np.sqrt(dx ** 2 + dy ** 2)

        d_size.append(r_circ / r_hit)

        if r_circ > 50:
            d_theta.append(solve_theta(x_circ, x_hit, y_circ, y_hit))

d_theta = np.array(d_theta)
d_theta = np.degrees(d_theta)
suggested_adj_theta = np.mean(d_theta) * human_learning_rate

suggested_adj_size = - (np.mean(d_size) - 1) * human_learning_rate

hit_error_adj_x = []
hit_error_adj_y = []

for i, circle in enumerate(assigned):
    if isinstance(circle, np.ndarray):
        theta = np.radians(suggested_adj_theta)
        hit_attempt = hit_attempts[i]

        x_hit = hit_attempt[1]
        y_hit = hit_attempt[2]

        x_circ = circle[1]
        y_circ = circle[2]

        x_hit_adj = (x_hit - 256) * np.cos(theta) - (y_hit - 192) * np.sin(theta) + 256
        y_hit_adj = (x_hit - 256) * np.sin(theta) + (y_hit - 192) * np.cos(theta) + 192

        hit_error_adj_x.append(x_circ - x_hit_adj)
        hit_error_adj_y.append(y_circ - y_hit_adj)

# print(d_theta)
print(f'mean theta deviation: {np.mean(d_theta)}\n'
      f'suggested adjustment: {suggested_adj_theta}')

print(f'mean size deviation: {np.mean(d_size)}\n'
      f'suggested adjustment: {suggested_adj_size}')

if plot:
    fig, ax = plt.subplots(figsize=(5, 5))

    circle = plt.Circle((0, 0), radius, color='C0', alpha=0.2)
    ax.add_patch(circle)
    ax.scatter(hit_error_x, hit_error_y, color='red', alpha=0.4, label='Original Hit Error Distribution')
    ax.scatter(hit_error_adj_x, hit_error_adj_y, color='purple', alpha=0.8, label=f'Suggested Adjustment {suggested_adj_theta:.2f} Degrees')
    ax.set_xlim(-radius - 20, radius + 20)
    ax.set_ylim(-radius - 20, radius + 20)
    plt.legend()
    plt.show()
