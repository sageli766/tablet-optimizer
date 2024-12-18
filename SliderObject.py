import math
def toVector2(pos):
    return Vector2(pos.x, pos.y)

def toVector2List(arr):
    new_arr = []
    for p in arr:
        new_arr.append(toVector2(p))
    return new_arr

class Vector2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

class DistanceTime:
    def __init__(self):
        self.distance = 0.0
        self.t = 0.0
        self.point = Vector2(0.0, 0.0)

class Curve:
    def __init__(self, curve_type):
        self.CurveType = curve_type
        self.Points = []
        self.CurveSnapshots = []
        self.PixelLength = float("inf")
        self.Length = -1
        self.Length = self.Length
        self.Linear = True if curve_type == 'Linear' else False

    def Init(self):
        self.Length = self.CalculateLength()

    def AddPoint(self, point: Vector2):
        self.Points.append(point)


    def Interpolate(self, t):
        raise NotImplementedError("Interpolate method in Bezier/Linear/Circle Class")


    def AddDistanceTime(self, distance, time, point: Vector2):
        dt = DistanceTime()
        dt.distance = distance
        dt.t = time
        dt.point = point
        self.CurveSnapshots.append(dt)


    def Lerp(self, a: Vector2, b: Vector2, t):
        return Vector2((1 - t) * a.x + t * b.x, (1 - t) * a.y + t * b.y)


    def Distance(self, a: Vector2, b: Vector2):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)



    def Atan2(self, a: Vector2):
        return math.atan2(a.y, a.x)


    def PositionAtDistance(self, d):
        high = len(self.CurveSnapshots) -1
        low = 0

        while low <= high:
            mid = int((high + low) / 2)
            if mid == high or mid == low:
                if mid + 1 >= len(self.CurveSnapshots):
                    return self.CurveSnapshots[mid].point
                else:
                    a = self.CurveSnapshots[mid]
                    b = self.CurveSnapshots[mid + 1]
                    return self.Lerp(a.point, b.point, (d - a.distance) / (b.distance - a.distance)) if (b.distance - a.distance) != 0 else a.point
            if self.CurveSnapshots[mid].distance > d:
                high = mid
            else:
                low = mid
        return Vector2(0.0, 0.0)


    def CalculateLength(self, prec = 0.1):
        self.AddDistanceTime(0.0, 0.0, Vector2(self.Points[0].x, self.Points[0].y))
        sum = 0
        if self.Linear:
            if len(self.Points) == 2:
                distance = self.Distance(self.Points[0], self.Points[1])
                if self.PixelLength > 0 and distance > self.PixelLength:
                    self.AddDistanceTime(self.PixelLength, 1, self.Lerp(self.Points[0], self.Points[1], self.PixelLength / distance))
                    distance = self.PixelLength
                else:
                    self.AddDistanceTime(distance, 1, self.Points[1])
                return distance
            else:
                return 0
        for f in [i * prec for i in range(int(1 / prec) + 1)]:
            if f > 1:
                f = 1
            fplus = f + prec
            if fplus > 1:
                fplus = 1
            a = self.Interpolate(f)
            b = self.Interpolate(fplus)
            distance = self.Distance(a, b)
            if sum == 0 or (self.PixelLength > 0 and distance + sum <= self.PixelLength):
                sum += distance
                self.AddDistanceTime(sum, fplus, b)
            else:
                break
        return sum


class Line(Curve):
    def __init__(self):
        super().__init__('Linear')

    def Interpolate(self, t):
        if len(self.Points) != 2:
            return Vector2(0.0, 0.0)
        return self.Lerp(self.Points[0], self.Points[1], t)

class Bezier(Curve):
    def __init__(self):
        super().__init__('Bezier')

    def AddPoint(self, point: Vector2):
        self.Points.append(point)
        if len(self.Points) == 2:
            self.Linear = True
        else:
            self.Linear = False

    def Interpolate(self, t):
        n = len(self.Points)
        if n == 2:
            return self.Lerp(self.Points[0], self.Points[1], t)

        pts = []
        for i in range(n):
            pts.append(Vector2(self.Points[i].x, self.Points[i].y))

        for k in range(1, n):
            for i in range(n - k):
                pts[i] = self.Lerp(pts[i], pts[i + 1], t)
        return pts[0]


class Circle(Curve):
    def __init__(self):
        super().__init__('Pass-Through')

    def AddPoint(self, point: Vector2):
        self.Points.append(point)
        if len(self.Points) != 3:
            self.Linear = True
        else:
            self.Linear = False

    def CircleCenter(self, A, B, C):
        a = Vector2((A.x + B.x) / 2, (A.y + B.y) / 2)
        u = Vector2(A.y - B.y, B.x - A.x)
        b = Vector2((B.x + C.x) / 2, (B.y + C.y) / 2)
        v = Vector2(B.y - C.y, C.x - B.x)
        d = Vector2(a.x - b.x, a.y - b.y)
        vu = v.x * u.y - v.y * u.x

        g = (d.x * u.y - d.y * u.x) / vu
        return Vector2(b.x + g * v.x, b.y + g * v.y)
        # a = Vector2((A.x + B.x) / 2, (A.y + B.y) / 2)
        # u = Vector2(A.y - B.y, B.x - A.x)
        # b = Vector2((B.x + C.x) / 2, (B.y + C.y) / 2)
        # v = Vector2(B.y - C.y, C.x - B.x)
        # d = Vector2(a.x - b.x, a.y - b.y)
        # vu = v.x * u.y - v.y * u.x
        # g = (d.x * u.y - d.y * u.x) / vu
        # return Vector2(b.x + g * v.x, b.y + g * v.y)



    def IsClockwise(self, a, b, c):
        return a.x * b.y - b.x * a.y + b.x * c.y - c.x * b.y + c.x * a.y - a.x * c.y > 0

    def collinear(self, a, b, c):
        x1, y1 = b.x - a.x, b.y - a.y
        x2, y2 = c.x - a.x, c.y - a.y
        return abs(x1 * y2 - x2 * y1) < 10 ** -12

    def Interpolate(self, t):
        if len(self.Points) == 3:

            if self.collinear(self.Points[0], self.Points[1], self.Points[2]):
                return self.Lerp(self.Points[0], self.Points[1], t)

            center = self.CircleCenter(self.Points[0], self.Points[1], self.Points[2])
            radius = self.Distance(self.Points[0], center)

            start = self.Atan2(Vector2(self.Points[0].x - center.x, self.Points[0].y - center.y))
            end = self.Atan2(Vector2(self.Points[2].x - center.x, self.Points[2].y - center.y))

            if self.IsClockwise(self.Points[0], self.Points[1], self.Points[2]):
                while end < start:
                    end += 2 * math.pi
            else:
                while start < end:
                    start += 2 * math.pi

            t = start + (end - start) * t
            temp = Vector2(math.cos(t) * radius, math.sin(t) * radius)
            return Vector2(temp.x + center.x, temp.y + center.y)

class SliderObject:
    def __init__(self, BaseLocation, StartTime, slider_type, Points, RepeatCount, PixelLength, duration, EndTime):
        self.BaseLocation = BaseLocation
        self.StartTime = StartTime
        self.type = slider_type
        self.Points = Points
        self.RepeatCount = RepeatCount
        self._PixelLength = PixelLength
        self.PixelLength = PixelLength
        self.duration = duration
        self.EndTime = EndTime
        self.TotalLength = -1
        self.TotalLength = -1
        self._SegmentEndTime = -1
        self.Curves = []
        #self.EndBaseLocation = self.PositionAtTime(1)


    def CreateCurve(self) -> Curve:
        if len(self.Points) == 0:
            return None
        elif len(self.Points) == 2:
            return Line()
        elif len(self.Points) > 3:
            return Bezier()
        if self.type == 'Linear':
            return Line()
        elif self.type == 'Bezier':
            return Bezier()
        elif self.type == 'Pass-Through':
            return Circle()
        else:
            return None

    def PositionAtTime(self, t):
        return self.PositionAtDistance(self.TotalLength * t)

    def PositionAtDistance(self, d):
        sum = 0
        for curve in self.Curves:
            if sum + curve.Length >= d:
                return curve.PositionAtDistance(d - sum)
            sum += curve.Length

        lastCurve = self.Curves[-1]
        return lastCurve.PositionAtDistance(d - (sum - lastCurve.Length))

    def CreateCurves(self):
        n = len(self.Points)
        if n == 0:
            return
        lastPoint = self.Points[0]
        currentCurve = None
        for i in range(n):
            if lastPoint.x == self.Points[i].x and lastPoint.y == self.Points[i].y:
                currentCurve = self.CreateCurve()
                self.Curves.append(currentCurve)
            currentCurve.AddPoint(self.Points[i])
            lastPoint = self.Points[i]

        self.TotalLength = 0
        lastIndex = len(self.Curves) - 1
        for i in range(lastIndex):
            self.Curves[i].Init()
            self.TotalLength += self.Curves[i].Length

        if lastIndex >= 0:
            lastCurve = self.Curves[lastIndex]
            lastCurve.PixelLength = self.PixelLength - self.TotalLength
            lastCurve.Init()
            self.TotalLength += lastCurve.Length



#
# if __name__ == '__main__':
#     # test = SliderObject(Vector2(75, 123), 530, "Linear", [Vector2(75, 123), Vector2(458, 245)], 1, 395.857142857143, 744, 1274)
#     # test.CreateCurves()
#     # val = test.PositionAtTime(1)
#     # print(f'{val.x} {val.y}')
#
#     map_name = "aaa.osu"
#     data = OsuFile(map_name).parse_file()
#     hit_objects = data.hit_objects
#     for obj in hit_objects:
#         if isinstance(obj, Slider):
#             points = obj.points
#             points.insert(0, obj.pos)
#             test = SliderObject(toVector2(obj.pos), obj.start_time, obj.curve_type, toVector2List(points),
#                                 obj.repeat_count, obj.pixel_length, obj.duration, obj.end_time)
#             test.CreateCurves()
#             val = test.PositionAtTime(1)
#             print(f'{val.x} {val.y}')

