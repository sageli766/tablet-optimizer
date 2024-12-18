from osrparse import Replay
from osupyparser import OsuFile
from osupyparser.osu.objects import Circle, Slider, Spinner
from osrparse.utils import Mod, Key
from stacking import stacking_fix

import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt

# print numpy arrays without scientific notation
np.set_printoptions(suppress=True)


def norm_2(x1, x2, y1, y2):
    '''
    Helper function to compute the 2-norm
    :param x1: x coordinate of the first point
    :param x2: x coordinate of the second point
    :param y1: y coordinate of the first point
    :param y2: y coordinate of the second point
    :return: Two-norm or Euclidean distance between the two vectors.
    '''
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def dist_from_center(x, y):
    '''
    Helper function to calculate the distance from the center of the playfield
    :param x: x coordinate of the point in consideration
    :param y: y coordinate of the point in consideration
    :return: The point's distance from the center of the playfield
    '''
    return norm_2(256, x, 192, y)


def solve_theta(x1, x2, y1, y2, degrees=True):
    '''
    Solves for the signed angle between two vectors: magnitude is done by computing the dot product, sign is taken from
    cross product
    :param x1: x coordinate of the first point
    :param x2: x coordinate of the second point
    :param y1: y coordinate of the first point
    :param y2: y coordinate of the second point
    :param degrees:
    :return: Signed angle between two vectors
    '''
    r_circle = np.array([x1 - 256, y1 - 192])
    r_hit = np.array([x2 - 256, y2 - 192])

    dot_prod = np.dot(r_circle, r_hit)
    cross_prod = r_circle[0] * r_hit[1] - r_circle[1] * r_hit[0]

    angle = np.arccos(dot_prod / (np.linalg.norm(r_circle) * np.linalg.norm(r_hit)))

    if cross_prod < 0:
        angle *= -1

    return angle

class HitDetector:
    '''
    Class for a hit analyzer object
    '''
    def __init__(self, replay=None, map=None, debug=False, human_learning_rate=1.5, ignore_radius=50):
        '''
        :param replay: relative or absolute path to the replay
        :param map: relative or absolute path to the map
        :param debug: print extra debug information to the console
        :param human_learning_rate: WIP- how fast people adjust to new tablet settings such as tilt
        :param ignore_radius: radius inside of which hits are not considered (causing possibly unwanted large tilts)
        '''
        self.replay = replay
        self.map = map
        self.debug = debug
        self.human_learning_rate = 1.5
        self.hits_array = []
        self.circles_array = []
        self.hit_errors = []
        self.ignore_radius = ignore_radius
        self.radius = 0
        self.adj_theta = 0
        self.adj_size = 0

    def set_map(self, map: str):
        if map.endswith('.osu'):
            self.map = map
            return 'Set Map'
        else:
            return 'Please select a valid osu map (.osu)'

    def set_replay(self, replay):
        if replay.endswith('.osr'):
            self.replay = replay
            return 'Set Replay'
        else:
            return 'Please select a valid osu replay (.osr)'

    def process_map_data(self):
        '''
        Processes map hits given replay and map path
        :return:
        '''
        # IMPORT REPLAY AND MAP
        r = Replay.from_path(self.replay)
        o_map = OsuFile(self.map).parse_file()

        r_data = r.replay_data
        m_data = o_map.hit_objects

        cs, od = o_map.cs, o_map.od

        # MAP STATS
        self.radius = 54.4 - 4.48 * cs
        window_300 = 80 - 6 * od
        window_100 = 140 - 8 * od
        window_50 = 200 - 10 * od

        hit_data = []
        map_data = []

        m_data = stacking_fix(m_data, o_map, self.radius, False)

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

        self.hit_errors = []

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
                    if hit_error <= self.radius:
                        self.hit_errors.append((x_circle - x_hit, y_circle - y_hit))
                        self.hits_array.append(hit_attempts[i])
                        self.circles_array.append(hit_circle)
                        if self.debug:
                            print(f'HIT time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
                        assigned[i] = hit_circle
                        miss = False
                        hit_count += 1
                        start_idx = i
                        break

                i += 1

            if miss and not isinstance(assigned[min_delay_idx], np.ndarray):
                x_hit = hit_attempts[min_delay_idx][1]
                y_hit = hit_attempts[min_delay_idx][2]
                self.hit_errors.append((x_circle - x_hit, y_circle - y_hit))
                self.hits_array.append(hit_attempts[min_delay_idx])
                self.circles_array.append(hit_circle)
                if self.debug:
                    print(f'MISS time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
                assigned[min_delay_idx] = hit_circle
                start_idx = min_delay_idx
                miss_count += 1

        if self.debug:
            print(f'Miss count: {miss_count}, Hit count: {hit_count}')


    def process_rotation(self):
        '''
        Calculates tilt of hits
        '''
        d_theta = []
        for hit_attempt, circle in zip(self.hits_array, self.circles_array):
            x_hit = hit_attempt[1]
            y_hit = hit_attempt[2]

            x_circ = circle[1]
            y_circ = circle[2]

            r_circ = dist_from_center(x_circ, y_circ)

            if r_circ > self.ignore_radius:
                d_theta.append(solve_theta(x_circ, x_hit, y_circ, y_hit))

            self.adj_theta = np.mean(d_theta) * self.human_learning_rate

        if self.debug:
            print(f'mean theta deviation: {np.degrees(np.mean(d_theta))}\n'
                  f'suggested adjustment: {np.degrees(self.adj_theta)}')

        return np.mean(d_theta)

    def process_size(self):
        '''
        Calculates size errors
        '''
        d_size = []

        for hit_attempt, circle in zip(self.hits_array, self.circles_array):
            x_hit = hit_attempt[1]
            y_hit = hit_attempt[2]

            x_circ = circle[1]
            y_circ = circle[2]

            r_circ = dist_from_center(x_circ, y_circ)
            r_hit = dist_from_center(x_hit, y_hit)

            d_size.append(r_hit / r_circ)

        self.adj_size = (np.mean(d_size) - 1)

        if self.debug:
            print(f'mean size deviation: {np.mean(d_size)}\n'
                  f'suggested adjustment: {self.adj_size}')

        return np.mean(d_size)

    def plot_hit_errors(self):
        '''
        Plots original hit errors against a hitcircle
        :return: figure, axis tuple of plot
        '''
        hit_error_x = [x for x, y in self.hit_errors]
        hit_error_y = [y for x, y in self.hit_errors]

        fig, ax = plt.subplots(figsize=(3, 3))

        circle = plt.Circle((0, 0), self.radius, color='C0', alpha=0.5)
        ax.add_patch(circle)
        ax.scatter(hit_error_x, hit_error_y, color='grey', alpha=0.5, label='Hits')
        ax.set_xlim(-self.radius - 20, self.radius + 20)
        ax.set_ylim(-self.radius - 20, self.radius + 20)
        fig.set_facecolor("#262626")
        ax.set_facecolor("#262626")
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        plt.legend(facecolor="#262626", labelcolor='white', loc='upper right')
        plt.title('Hit Error Distribution', color='white')
        if self.debug:
            plt.show()

        return fig, ax

    def plot_adj_hit_errors(self):
        '''
        Plots adjusted hit errors based on calculated tilt and size offsets
        :return: figure, axis objects of plot
        '''
        theta = - self.adj_theta
        hit_error_x = np.array([x for x, y in self.hit_errors])
        hit_error_y = np.array([y for x, y in self.hit_errors])

        x_hit = np.array([x for _, x, y, _ in self.hits_array])
        y_hit = np.array([y for _, x, y, _ in self.hits_array])

        x_circ = np.array([x for _, x, y in self.circles_array])
        y_circ = np.array([y for _, x, y in self.circles_array])

        # Rotate hits around center of playfield
        x_hit_adj = (x_hit - 256) * np.cos(theta) - (y_hit - 192) * np.sin(theta) + 256
        y_hit_adj = (x_hit - 256) * np.sin(theta) + (y_hit - 192) * np.cos(theta) + 192

        x_hit_adj *= 1 - self.adj_size
        y_hit_adj *= 1 - self.adj_size

        hit_error_adj_x = x_circ - x_hit_adj
        hit_error_adj_y = y_circ - y_hit_adj

        fig, ax = plt.subplots(figsize=(3, 3))

        circle = plt.Circle((0, 0), self.radius, color='C0', alpha=0.5)
        ax.add_patch(circle)
        ax.scatter(hit_error_x, hit_error_y, color='grey', alpha=0.5, label='Hits')
        ax.scatter(hit_error_adj_x, hit_error_adj_y, color='red', alpha=0.8, label='Adjusted')
        ax.set_xlim(-self.radius - 20, self.radius + 20)
        ax.set_ylim(-self.radius - 20, self.radius + 20)
        fig.set_facecolor("#262626")
        ax.set_facecolor("#262626")
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        plt.legend(facecolor="#262626", labelcolor='white', loc='upper right')
        plt.title('Adjusted Hit Errors', color='white')
        if self.debug:
            plt.show()

        return fig, ax

    def least_squares_fit(self):
        guess = [0., 0.]

        def residual(params):
            theta, size = params

            # hit location and circle location arrays
            x_hit = np.array([x for _, x, y, _ in self.hits_array])
            y_hit = np.array([y for _, x, y, _ in self.hits_array])

            x_circ = np.array([x for _, x, y in self.circles_array])
            y_circ = np.array([y for _, x, y in self.circles_array])

            # rotate hits around center of playfield
            x_hit_adj = (x_hit - 256) * np.cos(theta) - (y_hit - 192) * np.sin(theta) + 256
            y_hit_adj = (x_hit - 256) * np.sin(theta) + (y_hit - 192) * np.cos(theta) + 192

            # scale hit positions
            x_hit_adj *= size
            y_hit_adj *= size

            resid_x = x_circ - x_hit_adj
            resid_y = y_circ - y_hit_adj

            return np.concatenate((resid_x, resid_y))

        result = least_squares(residual, guess)

        theta_optimal, size_optimal = result.x

        if self.debug:
            print(f'[Least squares fit results] Theta: {np.degrees(theta_optimal)}, size: {size_optimal}')

        self.adj_theta = - theta_optimal
        self.adj_size = (1 - size_optimal)

        return theta_optimal, size_optimal


if __name__ == '__main__':
    test = HitDetector(r'.\replays_lobotomy\auto_BLOODY_RED_5deg.osr', r'.\maps\BLOODY_RED.osu', debug=True)
    test.process_map_data()
    # test.process_size()
    test.process_rotation()
    # test.plot_adj_hit_errors()
    test.least_squares_fit()
    test.plot_adj_hit_errors()
