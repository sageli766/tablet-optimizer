from osrparse import Replay
from osupyparser import OsuFile
from osupyparser.osu.objects import Circle, Slider, Spinner, Position
from osrparse.utils import Mod
import math
from SliderObject import toVector2, toVector2List, SliderObject, Vector2

class ObjectStack:
    def __init__(self,hit_object, stack_height):
        self.hit_object = hit_object
        self.stack_height = stack_height

def pos_distance(pos1, pos2):
    return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)


def stacking_fix(hit_objects, map, radius, enable_hardrock):
    if enable_hardrock:
        for i in range(len(hit_objects)):
            y = hit_objects[i].pos.y
            if y > 192:
                y = y + 2 * (192 - y)
            else:
                y = y - 2 * (192 - (384 - y))
            hit_objects[i].pos.y = y

            if isinstance(hit_objects[i], Slider):
                points = hit_objects[i].points
                for j in range(len(points)):
                    y = points[j].y
                    if y > 192:
                        y = y + 2 * (192 - y)
                    else:
                        y = y - 2 * (192 - (384 - y))
                    points[j].y = y
                hit_objects[i].points = points

    stack_objects = []
    for obj in hit_objects:

        if isinstance(obj, Slider):
            points = obj.points
            points.insert(0, obj.pos)
            test = SliderObject(toVector2(obj.pos), obj.start_time,
                                obj.curve_type, toVector2List(points),
                                obj.repeat_count, obj.pixel_length,
                                obj.duration, obj.end_time)
            test.CreateCurves()
            val = test.PositionAtTime(1)
            obj.end_position = Position(val.x, val.y)
        stack_objects.append(ObjectStack(obj, 0))

    ar_window = min(1800 - 120 * map.ar, 1950 - 150 * map.ar)
    stack_time_window = ar_window * (map.stack_leniency if map.stack_leniency else 7)
    stack_distance = 3

    for i in range(len(stack_objects) - 1, 0, -1):
        n = i
        objectI = stack_objects[i]

        if objectI.stack_height != 0 or isinstance(objectI.hit_object, Spinner):
            continue

        if isinstance(objectI.hit_object, Circle):
            while n - 1 >= 0:
                n -= 1
                objectN = stack_objects[n]
                if isinstance(objectN.hit_object, Spinner):
                    continue
                if isinstance(objectN.hit_object, Slider):
                    endTime = objectN.hit_object.end_time
                if isinstance(objectN.hit_object, Circle):
                    endTime = objectN.hit_object.start_time

                if objectI.hit_object.start_time - endTime > stack_time_window:
                    break

                if isinstance(objectN.hit_object, Slider) and pos_distance(objectN.hit_object.end_position, objectI.hit_object.pos) < stack_distance:

                    offset = objectI.stack_height - objectN.stack_height + 1

                    for j in range(n + 1, i + 1):
                        objectJ = stack_objects[j]
                        if pos_distance(objectN.hit_object.end_position, objectJ.hit_object.pos) < stack_distance:
                            objectJ.stack_height -= offset
                    break
                if pos_distance(objectN.hit_object.pos, objectI.hit_object.pos) < stack_distance:
                    objectN.stack_height = objectI.stack_height + 1
                    objectI = objectN

        elif isinstance(objectI.hit_object, Slider):
            while n - 1 >= 0:
                n -= 1
                objectN = stack_objects[n]
                if isinstance(objectN.hit_object, Spinner):
                    continue

                if objectI.hit_object.start_time - objectN.hit_object.start_time > stack_time_window:
                    break

                N_endPos = objectN.hit_object.end_position if isinstance(objectN.hit_object, Slider) else objectN.hit_object.pos
                if pos_distance(N_endPos, objectI.hit_object.pos) < stack_distance:
                    objectN.stack_height = objectI.stack_height + 1
                    objectI = objectN

    stack_offset = radius / 10
    all_hit_objects = []
    for stack_object in stack_objects:
        obj = stack_object.hit_object
        x = obj.pos.x - stack_offset * stack_object.stack_height
        y = obj.pos.y - stack_offset * stack_object.stack_height
        obj.pos.x = x
        obj.pos.y = y
        all_hit_objects.append(obj)

    return all_hit_objects