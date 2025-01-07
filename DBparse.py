"""
Parses osu!.db file following https://github.com/ppy/osu/wiki/Legacy-database-file-structure
"""

import struct

class DBparser:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_int(self, file):
        return struct.unpack("<i", file.read(4))[0]

    def read_bool(self, file):
        return struct.unpack("<?", file.read(1))[0]

    def read_short(self, file):
        return struct.unpack("<h", file.read(2))[0]

    def read_long(self, file):
        return struct.unpack("<q", file.read(8))[0]

    def read_single(self, file):
        return struct.unpack("<f", file.read(4))[0]

    def read_double(self, file):
        return struct.unpack("<d", file.read(8))[0]

    def read_byte(self, file):
        return struct.unpack("<b", file.read(1))[0]


    def read_string(self, file):
        marker = file.read(1)
        if marker == b'\x0b':  # Indicates the start of a string
            length = 0
            shift = 0
            while True:
                b = file.read(1)[0]
                length |= (b & 0x7F) << shift
                if b & 0x80 == 0:
                    break
                shift += 7
            return file.read(length).decode("utf-8")
        return ""

    def parse_beatmaps(self, file, count, osu_version):
        beatmaps = {}
        for _ in range(count):
            if osu_version < 20191106:
                size = self.read_int(file)  # Size in bytes (skip value)

            # read fields
            artist_name = self.read_string(file) # Artist name
            artist_unicode = self.read_string(file) # Artist unicode
            song_title = self.read_string(file) # Song title
            song_unicode = self.read_string(file) # Song unicode
            creator_name = self.read_string(file) # Creator name (mapper)
            difficulty = self.read_string(file) # Difficulty Name
            audio_file_name = self.read_string(file) # Audio file name associated with map
            md5_hash = self.read_string(file) # MD5 hash of map
            osu_file_name = self.read_string(file) # Name of osu file

            # skip over rest of fields
            ranked = self.read_bool(file)  # Ranked status
            hitcircle_count = self.read_short(file)
            slider_count = self.read_short(file)
            spinner_count = self.read_short(file)
            last_mod_time = self.read_long(file)

            if osu_version >= 20140609:
                AR = self.read_single(file)  # Approach rate
                CS = self.read_single(file)  # Circle size
                HP = self.read_single(file)  # HP drain
                OD = self.read_single(file)  # Overall difficulty
            else:
                AR = self.read_byte(file)  # Approach rate (legacy)
                CS = self.read_byte(file)  # Circle size (legacy)
                HP = self.read_byte(file)  # HP drain (legacy)
                OD = self.read_byte(file)  # Overall difficulty (legacy)

            SV = self.read_double(file)  # Slider velocity

            # TODO: extract star rating for mod combinations
            for _ in range(4):  # For each gamemode
                pair_count = self.read_int(file)
                for _ in range(pair_count):
                    file.read(14) # Size of Int-Double pair

            drain_time = self.read_int(file) # Drain time
            tot_time = self.read_int(file) # Total time
            preview_time = self.read_int(file) # Audio preview start time

            # TODO: extract timing points
            timing_points_count = self.read_int(file)
            for _ in range(timing_points_count):
                file.read(17)  # Timing points are 17 bytes long

            difficulty_id = self.read_int(file)  # Difficulty ID
            map_id = self.read_int(file)  # Beatmap ID
            thread_id =  self.read_int(file)  # Thread ID

            osu_grade = self.read_byte(file)  # Grade achieved in osu!
            taiko_grade = self.read_byte(file)  # Grade achieved in Taiko
            ctb_grade = self.read_byte(file)  # Grade achieved in CTB
            mania_grade = self.read_byte(file)  # Grade achieved in osu!mania

            offset = self.read_short(file)  # Local beatmap offset
            stack_leniancy = self.read_single(file) # Stack leniency (Single)
            gameplay_mode = self.read_byte(file)  # osu! gameplay mode

            # Additional beatmap fields
            song_source = self.read_string(file)  # Song source
            song_tags = self.read_string(file)  # Song tags
            online_offset = self.read_short(file) # Online offset
            font = self.read_string(file)  # Font used for the title of the song
            unplayed = self.read_bool(file) # Beatmap unplayed
            last_time_played = self.read_long(file)  # Last time when beatmap was played
            osz2 = self.read_bool(file) # Is the beatmap osz2
            folder_name = self.read_string(file)  # Folder name of the beatmap
            last_check = self.read_long(file)  # Last time checked against osu! repository

            ignore_sound = self.read_bool(file)  # Ignore beatmap sound
            ignore_skin = self.read_bool(file)  # Ignore beatmap skin
            disable_storyboard = self.read_bool(file)  # Disable storyboard
            disable_video = self.read_bool(file)  # Disable video
            vis_override = self.read_bool(file)  # Visual override

            if osu_version < 20140609:
                file.read(2)  # Unknown val

            last_mod_time_unknown = self.read_int(file)  # Last modification time (duplicate?)

            mania_scroll_speed = self.read_byte(file)  # Mania scroll speed

            # Create hashmap of md5 to file names
            beatmaps[md5_hash] = (folder_name, osu_file_name)

        return beatmaps

    def parse(self):
        with open(self.file_path, "rb") as file:
            osu_version = self.read_int(file)
            folder_count = self.read_int(file)
            account_unlocked = self.read_bool(file)

            # skip datetime field for now
            file.read(8)

            player_name = self.read_string(file)
            beatmap_count = self.read_int(file)
            beatmaps = self.parse_beatmaps(file, beatmap_count, osu_version)
            user_permissions = self.read_int(file)

            return {
                "osu_version": osu_version,
                "folder_count": folder_count,
                "account_unlocked": account_unlocked,
                "player_name": player_name,
                "beatmap_count": beatmap_count,
                "beatmaps": beatmaps,
                "user_permissions": user_permissions,
            }

if __name__ == '__main__':
    parser = DBparser("C:/Users/sagel/AppData/Local/osu!/osu!.db")
    osu_data = parser.parse()
    print(osu_data)
