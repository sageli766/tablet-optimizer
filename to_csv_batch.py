import os
from hitDetection import *

replay_dir = 'replays_various'
output_dir = 'data'
home_dir = 'C:/Users/sagel/AppData/Local/osu!'

for file in os.listdir(replay_dir):
    if file.endswith('.osr'):
        print(f'Now Processing {file}')
        try:
            replay_path = replay_dir + '/' + file
            map_path = map_path_from_replay(home_dir, replay_path)
            detector = HitDetector(replay_path, map_path)
            detector.process_map_data()
            detector.to_csv(f'{output_dir}/{file[:-4]}.csv')
        except:
            print(f'Error processings {file}')
