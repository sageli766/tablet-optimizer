import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import requests
from dotenv import load_dotenv
from ossapi import Ossapi

load_dotenv()

from osrparse import Replay
from osrparse.utils import Key, Mod
from osupyparser import OsuFile
from osupyparser.osu.objects import Spinner

from DBparse import DBparser
from stacking import stacking_fix

# print numpy arrays without scientific notation
np.set_printoptions(suppress=True)


def norm_2(x1, x2, y1, y2):
    """
    Helper function to compute the 2-norm
    :param x1: x coordinate of the first point
    :param x2: x coordinate of the second point
    :param y1: y coordinate of the first point
    :param y2: y coordinate of the second point
    :return: Two-norm or Euclidean distance between the two vectors.
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def dist_from_center(x, y):
    """
    Helper function to calculate the distance from the center of the playfield
    :param x: x coordinate of the point in consideration
    :param y: y coordinate of the point in consideration
    :return: The point's distance from the center of the playfield
    """
    return norm_2(256, x, 192, y)


def solve_theta(x1, x2, y1, y2, degrees=True):
    """
    Solves for the signed angle between two vectors: magnitude is done by computing the dot product, sign is taken from
    cross product
    :param x1: x coordinate of the first point
    :param x2: x coordinate of the second point
    :param y1: y coordinate of the first point
    :param y2: y coordinate of the second point
    :param degrees:
    :return: Signed angle between two vectors
    """
    r_circle = np.array([x1 - 256, y1 - 192])
    r_hit = np.array([x2 - 256, y2 - 192])

    dot_prod = np.dot(r_circle, r_hit)
    cross_prod = r_circle[0] * r_hit[1] - r_circle[1] * r_hit[0]

    angle = np.arccos(dot_prod / (np.linalg.norm(r_circle) * np.linalg.norm(r_hit)))

    if cross_prod < 0:
        angle *= -1

    return angle

def map_path_from_replay(home_dir, replay_path):
    parser = DBparser(f'{home_dir}/osu!.db')
    path_dict = parser.parse()['beatmaps']
    r = Replay.from_path(replay_path)
    map_hash = r.beatmap_hash
    map_dir = home_dir + '/Songs/' + path_dict[map_hash][0]
    map_path = map_dir + '/' + path_dict[map_hash][1]
    return map_path


class HitDetector:
    """
    Class for a hit analyzer object
    """
    def __init__(self, replay=None, map=None, debug=False, human_learning_rate=1, ignore_radius=50):
        """
        :param replay: relative or absolute path to the replay
        :param map: relative or absolute path to the map
        :param debug: print extra debug information to the console
        :param human_learning_rate: WIP- how fast people adjust to new tablet settings such as tilt
        :param ignore_radius: radius inside which hits are not considered (causing possibly unwanted large tilts)
        """
        self.replay = replay
        self.map = map
        self.debug = debug
        self.human_learning_rate = human_learning_rate
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
        """
        Processes map hits given replay and map path
        :return:
        """
        # Importing replay and map
        r = Replay.from_path(self.replay)
        o_map = OsuFile(self.map).parse_file()

        r_data = r.replay_data
        m_data = o_map.hit_objects

        # print(np.array(r_data[:10]))
        # print(np.array(m_data[:10]))

        cs, od = o_map.cs, o_map.od

        # Some map statistics
        self.radius = 54.4 - 4.48 * cs
        window_300 = 80 - 6 * od
        window_100 = 140 - 8 * od
        window_50 = 200 - 10 * od

        hit_data = []
        map_data = []

        m_data = stacking_fix(m_data, o_map, self.radius, False)

        # Convert replay and map data into numpy arrays for ease of use
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

        # Convert to numpy arrays
        hit_data = np.array(hit_data)
        map_data = np.array(map_data)

        # Change relative time to absolute time: cumulative sum across first column
        hit_data[:, 0] = np.cumsum(hit_data[:, 0])

        mask = [False]

        # TODO: Redo logic to be more concise
        # Filtering for valid keypress actions
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
        # mask_range = [0, 3000]
        # print(hit_data[(hit_data[:, 0] > mask_range[0]) & (hit_data[:, 0] < mask_range[1])])
        # print('\n')
        # print(hit_attempts[(hit_attempts[:, 0] > mask_range[0]) & (hit_attempts[:, 0] < mask_range[1])])

        self.hit_errors = []

        start_idx = 0
        miss_count = 0
        hit_count = 0
        hit_assigned = [False] * len(hit_attempts)

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
                if hit_attempts[i][0] > timing_lower and not isinstance(hit_assigned[i], np.ndarray):
                    # Find distance between hit attempt and hit circle
                    x_hit = hit_attempts[i][1]
                    y_hit = hit_attempts[i][2]
                    hit_error = norm_2(x_circle, x_hit, y_circle, y_hit)

                    # Store the closest attempt for later use
                    delay = abs(hit_circle[0] - hit_attempts[i][0])
                    if delay < min_delay:
                        min_delay_idx = i
                        min_delay = delay

                    # If the attempted hit is within the circle's radius,
                    # record and mark the hit attempt as used and continue
                    # and set new starting index as the location of the assigned hit
                    # We break the loop to choose the first unassigned action to the circle.
                    if hit_error <= self.radius:
                        self.hit_errors.append((x_circle - x_hit, y_circle - y_hit))
                        self.hits_array.append(hit_attempts[i])
                        self.circles_array.append(hit_circle)
                        if self.debug:
                            print(f'HIT time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
                        hit_assigned[i] = hit_circle
                        miss = False
                        hit_count += 1
                        start_idx = i
                        break

                i += 1

            # if no hit attempts are within the radius of the circle we choose the action closest in time to the circle
            if miss and not isinstance(hit_assigned[min_delay_idx], np.ndarray):
                x_hit = hit_attempts[min_delay_idx][1]
                y_hit = hit_attempts[min_delay_idx][2]
                self.hit_errors.append((x_circle - x_hit, y_circle - y_hit))
                self.hits_array.append(hit_attempts[min_delay_idx])
                self.circles_array.append(hit_circle)
                if self.debug:
                    print(f'MISS time:{time} at: ({x_hit: .2f}, {y_hit: .2f}) time: {hit_circle[0]} circle: ({x_circle: .2f}, {y_circle: .2f})')
                hit_assigned[min_delay_idx] = hit_circle
                start_idx = min_delay_idx
                miss_count += 1

        if self.debug:
            print(f'Miss count: {miss_count}, Hit count: {hit_count}')


    def process_rotation(self):
        """
        Calculates tilt of hits
        """
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
        """
        Calculates size errors
        """
        d_size = []

        for hit_attempt, circle in zip(self.hits_array, self.circles_array):
            x_hit = hit_attempt[1]
            y_hit = hit_attempt[2]

            x_circ = circle[1]
            y_circ = circle[2]

            r_circ = dist_from_center(x_circ, y_circ)
            r_hit = dist_from_center(x_hit, y_hit)

            if r_circ > self.ignore_radius:
                d_size.append(r_hit / r_circ)

        self.adj_size = (np.mean(d_size) - 1)

        if self.debug:
            print(f'mean size deviation: {np.mean(d_size)}\n'
                  f'suggested adjustment: {self.adj_size}')

        return np.mean(d_size)

    def plot_hit_errors(self):
        """
        Plots original hit errors against a hitcircle
        :return: figure, axis tuple of plot
        """
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
        """
        Plots adjusted hit errors based on calculated tilt and size offsets
        :return: figure, axis objects of plot
        """
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
        """
        Performs a least-squares fit to adjust hit locations relative to circle locations
        by optimizing rotation (theta) and scaling (size)
        :return: A tuple containing two elements:
            - theta_optimal (float): Optimal rotation value (in radians) for alignment.
            - size_optimal (float): Optimal scaling factor for size adjustment.
        :rtype: Tuple[float, float]
        """
        guess = [0., 1.]

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

    def generate_ideal_path(self, data_array):
        """
        Helper function for path error optimization method.
        Converts a numpy array [time, x, y] into an array of interpolated positions per millisecond.
        :param data_array: numpy array with shape [N, 3] where each row contains time, x, y
        :return: Array with structure [[time, x, y], ...] for each millisecond.
        """
        interpolated_array = []

        for i in range(len(data_array) - 1):
            start_time, start_x, start_y = data_array[i]
            end_time, end_x, end_y = data_array[i + 1]

            for t in range(int(start_time), int(end_time) + 1):
                factor = (t - start_time) / (end_time - start_time)
                inter_x = (1 - factor) * start_x + factor * end_x
                inter_y = (1 - factor) * start_y + factor * end_y
                interpolated_array.append([t, inter_x, inter_y])

        return np.array(interpolated_array)

    def filter_hit_data(self, hit_data, map_data):
        """
        Helper function for path error optimization method.
        Masks out rows in hit_data based on time gaps in map_data greater than 5 seconds.
        Ensures no duplicate time stamps are included in hit_data.
        :param hit_data: numpy array with shape [N, 4], where each row contains time, x, y, and action.
        :param map_data: numpy array with shape [N, 3], where each row contains time, x, and y.
        :return: Masked numpy array of hit_data.
        """
        _, unique_indices = np.unique(hit_data[:, 0], return_index=True)
        hit_data = hit_data[np.sort(unique_indices)]

        mask = np.zeros(len(hit_data), dtype=bool)
        for i in range(len(map_data) - 1):
            start_time, end_time = map_data[i, 0], map_data[i + 1, 0]
            if end_time - start_time > 1000:
                mask = mask | ((hit_data[:, 0] >= start_time) & (hit_data[:, 0] <= end_time))
        return hit_data[~mask]


    def path_error(self):
        r = Replay.from_path(self.replay)
        o_map = OsuFile(self.map).parse_file()

        r_data = r.replay_data
        m_data = o_map.hit_objects

        hit_data = []
        map_data = []

        m_data = stacking_fix(m_data, o_map, self.radius, False)

        # Convert replay and map data into numpy arrays for ease of use
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

        hit_data = np.array(hit_data)
        map_data = np.array(map_data)

        # change relative time to absolute time
        hit_data[:, 0] = np.cumsum(hit_data[:, 0])

        # filter hit_data based on time gaps in map_data
        hit_data = self.filter_hit_data(hit_data, map_data)
        ideal_path = self.generate_ideal_path(map_data)

        mask_hit = np.isin(hit_data[:, 0], ideal_path[:, 0])  # check if hit_data times exist in ideal_path
        hit_data = hit_data[mask_hit]

        mask_path = np.array([ideal_path[a, 0] in hit_data[:, 0] for a in range(len(ideal_path))])
        masked_path = ideal_path[mask_path]

        _, unique_indices = np.unique(masked_path[:, 0], return_index=True)
        masked_path = masked_path[np.sort(unique_indices)]

        if self.debug:
            print(hit_data)
            print(masked_path)

        guess = [0., 1.]

        def residual(params):
            theta, size = params

            x_hits = hit_data[:, 1]
            y_hits = hit_data[:, 2]

            x_path = masked_path[:, 1]
            y_path = masked_path[:, 2]

            # rotate hits around center of playfield
            x_hit_adj = (x_hits - 256) * np.cos(theta) - (y_hits - 192) * np.sin(theta) + 256
            y_hit_adj = (x_hits - 256) * np.sin(theta) + (y_hits - 192) * np.cos(theta) + 192

            # scale hit positions
            x_hit_adj *= size
            y_hit_adj *= size

            resid_x = x_path - x_hit_adj
            resid_y = y_path - y_hit_adj

            return np.concatenate((resid_x, resid_y))

        result = least_squares(residual, guess)

        theta_optimal, size_optimal = result.x

        if self.debug:
            print(f'[Path Error Lease Squares Results] Theta: {np.degrees(theta_optimal)}, size: {size_optimal}')

        self.adj_theta = - theta_optimal
        self.adj_size = (1 - size_optimal)

        return theta_optimal, size_optimal

    def get_osu_access_token(client_id, client_secret):
        token_url = "https://osu.ppy.sh/oauth/token"
        payload = {
            "client_id": client_id,
            'redirect_uri': 'https://google.com',
            "client_secret": client_secret,
            "grant_type": "client_credentials",
            "scope": "public"  # adjust the scopes as needed
        }
        response = requests.post(token_url, data=payload)
        response.raise_for_status()  # raises an error for bad responses
        token_info = response.json()
        return token_info["access_token"]

    def to_csv(self, path):
        if not self.hits_array or not self.circles_array:
            print('Please process replay before saving.')
            return

        home_dir = 'C:/Users/sagel/AppData/Local/osu!'

        r = Replay.from_path(self.replay)
        map_path = map_path_from_replay(home_dir, self.replay)
        o_map = OsuFile(map_path).parse_file()
        map_name = o_map.title
        player_id = r.username

        # Make .env file with your client id and client secrets
        client_id = os.getenv('CLIENT_ID')
        client_secret = os.getenv('CLIENT_SECRET')

        api = Ossapi(client_id, client_secret)

        # Extract player statistics like performance and accuracy
        player_stats = api.user(player_id).statistics

        performance = player_stats.pp
        acc = player_stats.hit_accuracy

        # Convert analyzer data into dataframes for ease of use
        objects = pd.DataFrame(self.circles_array, columns=['time', 'x', 'y'])

        # Subtraction of consecutive objects gives us distance and timing between circles
        objects_diff = objects - objects.shift(1)
        objects_diff.fillna(0, inplace=True)
        jump_distance = np.linalg.norm(objects_diff[['x', 'y']], axis=1)
        jump_delta_t = objects_diff['time']

        # Jump speed is defined as distance (game pixels) / time (ms) and has dimensionality units (game pixels) ms^-1
        if r.mods & Mod.DoubleTime:
            jump_delta_t *= 0.66
        elif r.mods & Mod.HalfTime:
            jump_delta_t *= 1.5
        jump_speed = jump_distance / jump_delta_t

        # Extract user hits from analyzer and compute error distances
        hits = pd.DataFrame(self.hits_array, columns=['time', 'x', 'y', 'action'])
        hit_errors_x = hits['x'] - objects['x']
        hit_errors_y = hits['y'] - objects['y']
        hit_errors = np.linalg.norm(np.vstack((hit_errors_x, hit_errors_y)), axis=0)

        if np.std(hit_errors) > 100:
            return 'Extremely high standard deviation in hit errors. Possible error with script.'

        # Create dataframe for export
        n_rows = len(hit_errors)
        data = pd.DataFrame({
            'player_id': [player_id] * n_rows,
            'player_pp': [performance] * n_rows,
            'player_acc': [acc] * n_rows,
            'map_id': [map_name] * n_rows,
            'error_distance': hit_errors,
            'jump_distance': jump_distance,
            'jump_delta_t': jump_delta_t,
            'jump_speed': jump_speed
        })

        data.to_csv(path, index=False)

if __name__ == '__main__':
    replay_path = r"replays_gmtn/razorfruit_gmtn.osr"
    home_dir = 'C:/Users/sagel/AppData/Local/osu!'
    map_path = map_path_from_replay(home_dir, replay_path)

    test = HitDetector(replay_path, map_path, debug=True)
    test.process_map_data()
    test.to_csv('test.csv')
    pass
    # test.path_error()
    # test.process_map_data()
    # test.process_rotation()
    # test.process_size()
    # test.least_squares_fit()
    # test.plot_adj_hit_errors()
