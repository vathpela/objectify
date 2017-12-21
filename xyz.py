#!/usr/bin/python3
""" xyz point coordinates """

# pylint: disable=no-self-use
# pylint: disable=too-many-arguments
# pylint: disable=too-many-lines
# pylint: disable=too-many-public-methods

import pdb
import math
import decimal
from daffine import Affine
from utility import Decimal

_Decimal = decimal.Decimal

def _inside(val, l, r):
    val = Decimal(val)
    if l < r:
        minimum = l
        maximum = r
    else:
        minimum = r
        maximum = l
    if val < minimum - _Decimal(0.01) or val > maximum + _Decimal(0.01):
        return False
    return True

def between(a, b, c, inclusive=True):
    " see if b is the middle"
    if inclusive:
        return b >= min(a, c) and \
                b <= max(a, c)
    return b > min(a, c) and \
            b < max(a, c)

class XYZ:
    "a point"
    __slots__ = ['_x', '_y', '_z', '_r', 'quant']
    def __init__(self, x, y, z=None, r=0, quant="10000.000000"):
        self.quant = quant
        self._x = Decimal(x, quant=quant)
        self._y = Decimal(y, quant=quant)
        self._z = Decimal(z, quant=quant)
        self._r = Decimal(r, quant=quant)

    @property
    def x(self):
        "x"
        return Decimal(self._x, quant=self.quant)

    @property
    def y(self):
        "y"
        return Decimal(self._y, quant=self.quant)

    @property
    def z(self):
        "z"
        if self._z is None:
            raise AttributeError('z')
        return Decimal(self._z, quant=self.quant)

    @property
    def r(self):
        "r"
        return Decimal(self._r, quant=self.quant)

    @property
    def xy(self):
        "xy"
        return XY(self.x, self.y)

    @property
    def xz(self):
        "xz"
        return XY(self.x, self.z)

    @property
    def yz(self):
        "yz"
        return XY(self.y, self.z)

    @property
    def xyquadrant(self):
        "quadrant on xy"
        return self._quadrant(self.x, self.y)

    @property
    def yzquadrant(self):
        "quadrant on yz"
        return self._quadrant(self.y, self.z)

    @property
    def xzquadrant(self):
        "quadrant on xz"
        return self._quadrant(self.x, self.z)

    @property
    def quadrant(self):
        "quadrant"
        return self.xyquadrant

    def __lt__(self, other):
        if self._z:
            origin = XYZ(0, 0, 0)
        else:
            origin = XYZ(0, 0)
        sd = self.distance(origin)
        od = other.distance(origin)
        return sd < od

    def __add__(self, other):
        return XYZ(self.x + other.x, self.y + other.y, other.z)

    def __sub__(self, other):
        return XYZ(self.x - other.x, self.y - other.y, self.z)

    def __str__(self):
        if self._z:
            fmt = "XYZ("
        else:
            fmt = "XY("
        if int(self.x) == _Decimal(self.x):
            fmt += "%d,"
        else:
            fmt += "%s,"
        if int(self.y) == _Decimal(self.y):
            fmt += "%d"
        else:
            fmt += "%s"
        if self._z:
            if int(self.z) == _Decimal(self.z):
                fmt += ",%d)"
            else:
                fmt += ",%s)"
            return fmt % (self.x, self.y, self.z)
        fmt += ")"
        return fmt % (self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if self._z is not None and not hasattr(other, 'z'):
            return False
        if self._z is None and hasattr(other, 'z'):
            return False
        if self._z is not None and hasattr(other, 'z'):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return self.x == other.x and self.y == other.y

    def _slope(self, x0, y0, x1, y1):
        "compute slope"
        ret = (y1 - y0) / (x1 - x0)
        return ret

    def slope(self, other):
        "default slope (xy axis)"
        return self.xyslope(other)

    def xyslope(self, other):
        "xyslope"
        return self._slope(self.x, self.y, other.x, other.y)

    def xzslope(self, other):
        "xzslope"
        return self._slope(self.x, self.z, other.x, other.z)

    def yzslope(self, other):
        "yzslope"
        return self._slope(self.y, self.z, other.y, other.z)

    def leftof(self, other):
        "leftof"
        return self.x < other.x

    def rightof(self, other):
        "rightof"
        return self.x > other.x

    def farther(self, other):
        "farther"
        return self.y > other.y

    def nearer(self, other):
        "nearer"
        return self.y < other.y

    def above(self, other):
        "above"
        return self.z > other.z

    def below(self, other):
        "below"
        return self.z < other.z

    def xyplanar(self, other):
        "on the same xy plane"
        return self.z == other.z

    def xzplanar(self, other):
        "on the same xz plane"
        return self.y == other.y

    def yzplanar(self, other):
        "on the same yz plane"
        return self.x == other.x

    def xlinear(self, other):
        "on the same x line"
        return self.y == other.y and self.z == other.z

    def ylinear(self, other):
        "on the same y line"
        return self.x == other.x and self.z == other.z

    def zlinear(self, other):
        "on the same z line"
        return self.x == other.x and self.y == other.y

    def __contains__(self, key):
        if self._z:
            return key in ['x', 'y', 'z']
        return key in ['x', 'y']

    def keys(self):
        "keys"
        yield 'x'
        yield 'y'
        if self._z:
            yield 'z'

    def __getitem__(self, key):
        indices = {0: self.x, 1: self.y, 2: self.z}
        keys = {'x':self.x, 'y':self.y, 'z':self.z}
        if self._z is not None:
            del indices[2]
            del keys['z']
        try:
            if isinstance(key, int):
                return indices[key]
            return keys[key]
        except KeyError:
            raise IndexError(key)

    def distance(self, other):
        "distance"
        x = other.x - self.x
        x *= x
        x = Decimal(x, quant=self.quant)
        y = other.y - self.y
        y *= y
        y = Decimal(y, quant=self.quant)
        try:
            otherz = other.z
        except AttributeError:
            otherz = self._z
        if not self._z:
            return Decimal(math.sqrt(x+y), quant=self.quant)

        z2 = otherz
        z = z2 - self.z
        z *= z
        z = Decimal(z, quant=self.quant)
        return Decimal(math.sqrt(x+y+z), quant=self.quant)

    def __hash__(self):
        return hash((self.x, self.y, self.z, self.r))

    def _quadrant(self, x, y):
        " figure out a quadrant based on x and y "
        if x > 0:
            if y > 0:
                return 1
            elif y == 0:
                raise ValueError
            return 4
        elif x < 0:
            if y > 0:
                return 2
            elif y == 0:
                raise ValueError
            return 3
        raise ValueError

class LineView:
    "ways to look at a line"

    def __init__(self, line, axis):
        self.line = line

        axis_methods = {
            'xy': {"xmax":"xmax",
                   "xmin":"xmin",
                   "ymax":"ymax",
                   "ymin":"ymin"},
            'xz': {"xmax":"xmax",
                   "xmin":"xmin",
                   "ymax":"zmax",
                   "ymin":"zmin"},
            'yz': {"xmax":"ymax",
                   "xmin":"ymin",
                   "ymax":"zmax",
                   "ymin":"zmin"}
        }
        self.methods = axis_methods[axis]

    @property
    def xmax(self):
        "xmax"
        return getattr(self.line, self.methods["xmax"])

    @property
    def xmin(self):
        "xmin"
        return getattr(self.line, self.methods["xmin"])

    @property
    def ymax(self):
        "ymax"
        return getattr(self.line, self.methods["ymax"])

    @property
    def ymin(self):
        "ymin"
        return getattr(self.line, self.methods["ymin"])

    # rise is in distance units
    @property
    def rise(self):
        "rise"
        return self.ymax - self.ymin

    # run is in distance units
    @property
    def run(self):
        "run"
        return self.xmax - self.xmin

    @property
    def length(self):
        "length"
        return self.xy_min.distance(self.xy_max)

    @property
    def xy_min(self):
        "xy maximum"
        return self.line.xy_min

    @property
    def xy_max(self):
        "xy minimum"
        return self.line.xy_max

    # m is a unitless ratio
    @property
    def m(self):
        "slope"
        try:
            return Decimal(self.rise / self.run, quant=self.line.quant)
        except decimal.DivisionByZero:
            return math.inf

    @property
    def b(self):
        "the b in 'y = mx + b'"
        # y = mx + b
        # y - mx = b
        # b = y - mx
        ret = self.ymax - (self.m * self.xmax)
        return Decimal(ret, quant=self.line.quant)

    # theta is in degrees
    @property
    def theta(self):
        "theta in degrees"
        atan = math.atan2(self.rise, self.run)
        return math.degrees(atan)

    # sin/cos/tan/etc are all ratios of distance units
    @property
    def sin(self):
        "sine"
        return math.sin(math.radians(self.theta)) * self.length

    @property
    def cos(self):
        "cosine"
        return math.cos(math.radians(self.theta)) * self.length

    @property
    def tan(self):
        "tangent"
        return math.tan(math.radians(self.theta)) * self.length

    @property
    def cotan(self):
        "cotangent"
        return (self.cos / self.sin) * self.length

    @property
    def sec(self):
        "secant"
        return math.hypot(self.length, self.tan)

    @property
    def cosec(self):
        "cosecant"
        return (self.length / self.sin) * self.length

    @property
    def versin(self):
        "versine"
        return self.length - self.cos

    @property
    def exsec(self):
        "exsecant"
        return self.sec - self.cos - self.versin

    @property
    def crd(self):
        "chord"
        return math.hypot(self.versin, self.sin)

    @property
    def cvs(self):
        "coversine"
        return self.length - self.sin

    @property
    def excosec(self):
        "excosecant"
        return self.cosec - self.sin - self.cvs

    def translate(self, xoff, yoff):
        "apply an affine translation"
        tlate = Affine.translation(xoff, yoff)

        p0 = XYZ(*self.xy_min * tlate)
        p1 = XYZ(*self.xy_max * tlate)

        return Line(p0, p1)

    def scale(self, xscale, yscale):
        "apply an affine scale"
        scale = Affine.scale(xscale, yscale)

        p0 = XYZ(*self.xy_min * scale)
        p1 = XYZ(*self.xy_max * scale)

        return Line(p0, p1)

    def rotate(self, angle, pivot=None):
        "apply an affine rotation around the or the supplied pivot point"
        rotator = Affine.rotation2d(angle, pivot)

        p0 = XYZ(*self.xy_min * rotator)
        p1 = XYZ(*self.xy_max * rotator)

        return Line(p0, p1)

    def YAtX(self, x):
        "y at x"
        return self.m * Decimal(x, quant=self.line.quant) + self.b

    def XAtY(self, y):
        "x at y"
        # y = mx + b
        # y -b = mx
        # (y-b)/m = x
        return (Decimal(y, quant=self.line.quant) - self.b) / self.m

    def __contains__(self, point):
        r = point.r
        if point.r == 0:
            # dude, whatever
            r = 0.5
        r = Decimal(r, quant=self.line.quant)

        if r < point.distance(self.xy_max):
            return True
        if r < point.distance(self.xy_min):
            return True

        line0 = self.translate(self.rise * r, - self.run * r)
        line1 = self.translate(- self.rise * r, self.run * r)

        line0 = line0.xy.rotate(self.line.theta, self.line.xy_min)
        line1 = line0.xy.rotate(self.line.theta, self.line.xy_min)

        rotator = Affine.rotation2d(self.line.theta, self.line.xy_min)
        rpoint = point * rotator

        return between(line0.xy_min.x, rpoint.x, line0.xy_max.x) and \
                between(min(line0.xy_min.y,
                            line0.xy_max.y,
                            line1.xy_min.y,
                            line1.xy_max.y),
                        rpoint.y,
                        max(line0.xy_min.y,
                            line0.xy_max.y,
                            line1.xy_min.y,
                            line1.xy_max.y))

class Line(object):
    "A line"
    _strname = "Line"

    def __init__(self, xy_min, xy_max, quant=None):
        self.xy_min = xy_min
        self.xy_max = xy_max
        self.color = None
        if quant:
            self.quant = quant
        else:
            self.quant = xy_min.quant

    def __str__(self):
        return "%s(%s,%s)" % (self._strname, self.xy_min, self.xy_max)

    def __repr__(self):
        return str(self)

    @property
    def length(self):
        "length"
        return self.xy_min.distance(self.xy_max)

    def __len__(self):
        return self.length

    def __eq__(self, other):
        if self.length == other.length:
            return True
        return False

    def __lt__(self, other):
        if self.length < other.length:
            return True
        return False

    def __hash__(self):
        return hash((self.xy_min, self.xy_max))

    @property
    def points(self):
        "points"
        yield self.xy_min
        yield self.xy_max

    @property
    def middle(self):
        "middle"
        x = self.xmin + (self.xmax - self.xmin) / _Decimal(2.0)
        x = Decimal(x, quant=self.quant)
        y = self.ymin + (self.ymax - self.ymin) / _Decimal(2.0)
        y = Decimal(y, quant=self.quant)
        try:
            z = self.zmin + (self.zmax - self.zmin) / _Decimal(2.0)
            z = Decimal(z, quant=self.quant)
            return XY(x, y, z)
        except AttributeError:
            return XY(x, y)

    @property
    def bottom(self):
        "bottom"
        return Line(self.xy_min, self.middle)

    @property
    def top(self):
        "top"
        return Line(self.middle, self.xy_max)

    @property
    def xmin(self):
        "xmin"
        return self.xy_min.x
    @property
    def xmax(self):
        "xmax"
        return self.xy_max.x

    @property
    def ymin(self):
        "ymin"
        return self.xy_min.y
    @property
    def ymax(self):
        "ymax"
        return self.xy_max.y

    @property
    def zmin(self):
        "zmin"
        try:
            return self.xy_min.z
        except AttributeError:
            raise AttributeError("zmin")
    @property
    def zmax(self):
        "zmax"
        try:
            return self.xy_max.z
        except AttributeError:
            raise AttributeError("zmax")

    @property
    def xrange(self):
        "xrange"
        return self.xmax - self.xmin
    @property
    def yrange(self):
        "yrange"
        return self.ymax - self.ymin
    @property
    def zrange(self):
        "zrange"
        try:
            return self.zmax - self.zmin
        except AttributeError:
            raise AttributeError("zrange")

    @property
    def xy(self):
        "xy line view"
        return LineView(self, "xy")

    @property
    def xz(self):
        "xz line view"
        return LineView(self, "xz")

    @property
    def yz(self):
        "yz line view"
        return LineView(self, "yz")

    def __contains__(self, point):
        try:
            return point in self.xy and point in self.xz and point in self.yz
        except AttributeError:
            return point in self.xy

    def crosses(self, axis, position):
        "test if this line exists on the specified axis at position"
        pmin, pmax = [getattr(self, attr) for attr in \
                      {'x': ['xmin', 'xmax'],
                       'y': ['ymin', 'ymax'],
                       'z': ['zmin', 'zmax']}[axis]]
        return between(pmin, position, pmax)

    def crossesX(self, x):
        "test if this line exists on the x axis at x"
        return between(self.xmin, x, self.xmax)

    def crossesY(self, y):
        "test if this line eyists on the y axis at y"
        return between(self.ymin, y, self.ymax)

    def crossesZ(self, z):
        "test if this line ezists on the z axis at z"
        return between(self.zmin, z, self.zmax)

    def atD(self, d):
        "get this line's point at distance d from xy_min"
        t = self.length / d
        x0 = self.xy_min.x
        x1 = self.xy_max.x
        dx = x1 - x0

        y0 = self.xy_min.y
        y1 = self.xy_max.y
        dy = y1 - y0

        x = x0 + dx / t
        y = y0 + dy / t

        if hasattr(self.xy_min, 'z') and hasattr(self.xy_max, 'z'):
            z0 = self.xy_min.z
            z1 = self.xy_max.z
            dz = z1 - z0
            if d < 0:
                dz = 0 - abs(dz)
            z = z0 + dz / t
            return XYZ(x, y, z, quant=self.quant)
        return XYZ(x, y, quant=self.quant)

    def atX(self, x):
        "the point where this line intersects the given x"
        if not between(self.xmin, x, self.xmax):
            return None
        return self.atD(x - self.xmin)

    def atY(self, y):
        "the point where this line intersects the given y"
        if not between(self.ymin, y, self.ymax):
            return None
        return self.atD(y - self.ymin)

    def atZ(self, z):
        "the point where this line intersects the given z"
        if not between(self.zmin, z, self.zmax):
            return None
        return self.atD(z - self.zmin)

    @property
    def reverse(self):
        "put the start the other way around"
        line = Line(self.xy_max, self.xy_min)
        line.color = self.color
        return line

    def distance(self, point):
        "calculate the minimum distance from the line to a point"
        d = self.xy_min.distance(point) + self.xy_max.distance(point)
        d = _Decimal(d) / 2
        return Decimal(d)

    @property
    def xybisector(self):
        "build a bisecting line to this one along xy"
        m = self.middle

        rise = self.xy.rise
        run = self.xy.run

        point1 = XY(m.x - rise/2, m.y + run/2)
        l = Line(m, point1)
        point0 = l.atD(0 - (self.length/2))
        l = Line(point0, m)
        point1 = l.atD(self.length)
        l = Line(point0, point1)

        if hasattr(self.middle, 'z'):
            point0 = XYZ(point0.x, point0.y, self.middle.z)
            point1 = XYZ(point1.x, point1.y, self.middle.z)
        return Line(point0, point1)

class Index:
    """An indexed array for anything that's not a number"""

    __slots__ = ['itemlist', 'indices', 'n', 'start', 'aliases']

    def __init__(self, start=0):
        self.itemlist = []
        self.indices = {}
        self.n = - 1
        self.start = start
        self.aliases = {}

    def __getitem__(self, key):
        if not isinstance(key, int):
            key = self.indices[key] + self.start
        return self.itemlist[key - self.start]

    def __contains__(self, key):
        if not isinstance(key, int):
            key = self.indices[key] + self.start
        key -= self.start
        return key >= 0 and key <= self.n

    def __len__(self):
        return self.n - 1 - self.start

    def __iter__(self):
        for item in self.itemlist:
            yield item

    def append(self, x):
        "Append"
        assert not isinstance(x, int)
        entry = self.indices.setdefault(x, self.n+1)
        assert isinstance(entry, int)
        if entry > self.n:
            self.n += 1
            self.itemlist.append(x)
            assert self.itemlist[entry] == x
        else:
            self.n += 1
            self.itemlist.append(self.itemlist[entry])
            self.aliases.setdefault(self.n, [])
            self.aliases[self.n].append(entry)
        return entry + self.start

    def index(self, x):
        "Get the index of a member"
        return self.indices[x] + self.start

    def count(self, x):
        "count occurances of a member"
        if x in self:
            return 1
        return 0

class Vertex(XYZ):
    """A vertex belonging to a vertex library"""

    __slots__ = ['library', '_x', '_y', '_z', 'index', 'faces']

    def __init__(self, library, x, y, z=None):
        XYZ.__init__(self, x, y, z)
        assert isinstance(library, Index)
        self.library = library
        self.faces = []
        self.index = library.append(self)

    def __str__(self):
        fmt = "Vertex("
        if int(self.x) == _Decimal(self.x):
            fmt += "%d,"
        else:
            fmt += "%s,"
        if int(self.y) == _Decimal(self.y):
            fmt += "%d"
        else:
            fmt += "%s"
        if self._z:
            if int(self.z) == _Decimal(self.z):
                fmt += ",%d)"
            else:
                fmt += ",%s)"
            return fmt % (self.x, self.y, self.z)
        fmt += ")"
        return fmt % (self.x, self.y)

class Face(object):
    """A single face, normalized to have no more than 3 vertices"""

    __slots__ = ['library', 'i0', 'i1', 'i2']

    def __init__(self, library: Index, *vertices: [int]):
        assert isinstance(library, Index)
        self.library = library
        assert len(vertices) == 3
        assert isinstance(vertices[0], int)
        try:
            assert vertices[0] in library
        except AssertionError:
            raise AssertionError("%s in library" % (vertices[0],))
        self.i0 = vertices[0]
        self.v0.faces.append(self)
        assert isinstance(vertices[1], int)
        try:
            assert vertices[1] in library
        except AssertionError:
            raise AssertionError("%s in library" % (vertices[1],))
        self.i1 = vertices[1]
        self.v1.faces.append(self)
        assert isinstance(vertices[2], int)
        try:
            assert vertices[2] in library
        except AssertionError:
            raise AssertionError("%s in library" % (vertices[2],))
        self.i2 = vertices[2]
        self.v2.faces.append(self)

    def __new__(cls):
        obj = object.__new__(cls)
        return obj

    @classmethod
    def makeFromVertices(cls, library: Index, *vertices: [Vertex]):
        "normalize a set of points into a series of 3-sided faces"""
        assert isinstance(library, Index)
        vertices = list(vertices)
        vertices.sort()
        while len(vertices) >= 3:
            first = vertices[0:3]
            obj = cls.__new__(cls)
            obj.__init__(library, *first)
            newverts = list(vertices[3:])
            if len(newverts) not in (0, 3):
                for x in range(len(newverts) - 3, 0, 1):
                    newverts.insert(0, vertices[3 + x])
            vertices = newverts
            yield obj

    def split(self, axis, position):
        "split this face into multiple faces at position on axis"
        crossers = []
        above = []
        below = []
        pmin, pmax = [getattr(self, attr) for attr in \
                      {'x': ['xmin', 'xmax'],
                       'y': ['ymin', 'ymax'],
                       'z': ['zmin', 'zmax']}[axis]]
        for line in self.lines:
            if line.crosses(axis, position):
                crossers.append(line)
            elif pmin < position:
                above.append(line)
            elif pmax > position:
                below.append(line)
            else:
                # the line is already on this position
                return self.lines
        # XXX find break the lines that cross it in two, and add a line
        # between them that's on the specified position

    @property
    def v0(self):
        "vertex 0"
        ret = self.library[self.i0]
        return ret

    @property
    def v1(self):
        "vertex 1"
        return self.library[self.i1]

    @property
    def v2(self):
        "vertex 2"
        return self.library[self.i2]

    @property
    def vertices(self):
        """the vertices of this face"""
        return [self.v0, self.v1, self.v2]

    def keys(self):
        "keys"
        yield 'v0'
        yield 'v1'
        yield 'v2'

    def __getitem__(self, key):
        indices = {0: self.v0, 1: self.v1, 2: self.v2}
        keys = {'v0':self.v0, 'v1':self.v1, 'v2':self.v2}
        try:
            if isinstance(key, int):
                v = indices[key]
            else:
                v = keys[key]
            return self.library[v]
        except KeyError:
            raise IndexError(key)

    def __str__(self):
        verts = ",".join([str(v) for v in self.vertices])
        s = "Face(%s)" % (verts,)
        return s

    def __repr__(self):
        return str(self)

    def zintersection(self, z: _Decimal):
        """return the line at z or none if this doesn't cross it"""
        min_z = min([v.z for v in self.vertices])
        max_z = max([v.z for v in self.vertices])
        if min_z > z or max_z < z:
            return None
        # I don't know why this part was here:
        #min_x = min([v.x for v in self.vertices])
        #max_x = max([v.x for v in self.vertices])
        #min_y = min([v.y for v in self.vertices])
        #max_y = max([v.y for v in self.vertices])
        #pts = [XYZ(min_x, min_y, z),
        #       XYZ(max_x, max_y, z),
        #       XYZ(min_x, max_y, z)]

        l = len(self.vertices)
        return XY(sum([x.x for x in self.vertices]) / l, \
                  sum([x.y for x in self.vertices]) / l)

    @property
    def zline(self):
        "draw a line at z"
        return Line(XY(self.xmin, self.ymin), XY(self.xmax, self.ymax))

    @property
    def xmax(self):
        "maximum x"
        return max([v.x for v in self.vertices])

    @property
    def xmin(self):
        "minimum x"
        return min([v.x for v in self.vertices])

    def withinX(self, left: _Decimal, right: _Decimal):
        "are left and right within x?"
        return left >= self.xmin and right <= self.xmax

    @property
    def ymax(self):
        "maximum y of the vertices"
        return max([v.y for v in self.vertices])

    @property
    def ymin(self):
        "minimum y of the vertices"
        return min([v.y for v in self.vertices])

    def withinY(self, front: _Decimal, back: _Decimal):
        "are left and right within y?"
        return front >= self.ymin and back <= self.ymax

    @property
    def zmax(self):
        "maximum z of the vertices"
        return max([v.z for v in self.vertices])

    @property
    def zmin(self):
        "minimum z of the vertices"
        return min([v.z for v in self.vertices])

    @property
    def zavg(self):
        "average z of the vertices"
        return sum([v.z for v in self.vertices]) / len(self.vertices)

    def __lt__(self, other):
        if self.zavg == other.zavg:
            if self.zmin > other.zmin:
                return False
            return True
        if self.zavg > other.zavg:
            return False
        return True

    def __gt__(self, other):
        if self.zavg == other.zavg:
            if self.zmin < other.zmin:
                return False
            return True
        if self.zavg < other.zavg:
            return False
        return True

    def __eq__(self, other):
        if self.zavg != other.zavg:
            return False
        if self.zmin != other.zmin:
            return False
        if self.zmax != other.zmax:
            return False
        return True

    def __hash__(self):
        return hash(tuple(self.vertices))

    def crosses(self, axis, position):
        "crosses axis at offset position from origin"
        for line in self.lines:
            if line.crosses(axis, position):
                return True
        return False

    @property
    def lines(self):
        "the lines of this face"
        l = len(self.vertices) - 1
        for i, j in [(x, x+1) for x in range(0, l)] + [(l, 0)]:
            v0 = self.vertices[i]
            v1 = self.vertices[j]
            pointa = XYZ(v0.x, v0.y, v0.z)
            pointb = XYZ(v1.x, v1.y, v1.z)
            yield Line(pointa, pointb)

    def lineAtZ(self, z):
        "the line across this face at a given z"
        crossers = []
        points = []

        indent = "  lineAtZ:"

        for line in self.lines:
            if line.crossesZ(z):
                crossers.append(line)
                # print("%s %s crosses %f" % (indent, line, z))

        for line in crossers:
            if line.xy_min.z == z or line.xy_max.z == z:
                if line.xy_min.z == z:
                    point = line.xy_min
                    print("%s point: %s" % (indent, point))
                    points.append(point)

                if line.xy_max.z == z:
                    if not line.xy_max in points:
                        point = line.xy_max
                        print("%s point: %s" % (indent, point))
                        points.append(point)
                crossers.remove(line)

        if 2 - len(points) != len(crossers):
            pdb.set_trace()
            raise RuntimeError("mismatched points (%d) vs crossers (%d)" %
                               (2 - len(points), len(crossers)))

        # slopes=[c.xym for c in crossers]
        for crosser in crossers:
            # print("%s crosser: %s" % (indent, crosser))
            point = crosser.atZ(z)
            # print("%s point: %s" % (indent, point))
            points.append(point)

        if len(points) < 2:
            pdb.set_trace()
            raise ValueError

        line = Line(points[0], points[1])
        # print("%s new line %s (slope %s)" % (indent, line, line.xym))
        return line

class Model:
    "A thing built of faces"

    __slots__ = ['vertices', 'faces']

    def __init__(self):
        self.vertices = Index(start=1)
        self.faces = set()

    def __getstate__(self):
        return {'vertices':[(float(v.x), float(v.y), float(v.z)) for v in
                            self.vertices],
                'faces': [(face.i0, face.i1, face.i2) for face in self.faces]
               }

    def __setstate__(self, state):
        self.__init__()
        for vertex in state['vertices']:
            self.addVertex([Decimal(str(p)) for p in vertex])
        for vertices in state['faces']:
            self.addFace(*vertices)

    def addVertex(self, vertex: []):
        "add a vertex"
        vertex = Vertex(self.vertices, *vertex)
        try:
            return self.vertices[vertex]
        except KeyError:
            return self.vertices.append(vertex)

    def addFace(self, *vertices: [int]):
        "add a face"
        for face in Face.makeFromVertices(self.vertices, *vertices):
            self.faces.add(face)

    def xSlice(self, x: _Decimal):
        "slice at x"
        new = Model()
        for face in self.faces:
            if face.crossesX(x):
                vertices = []
                for v in face.vertices:
                    i = new.addVertex(v)
                    vertices.append(self.vertices[i])
                new.addFace(*vertices)
        return new

    def ySlice(self, y: _Decimal):
        "slice at y"
        new = Model()
        for face in self.faces:
            if face.crossesY(y):
                vertices = []
                for v in face.vertices:
                    i = new.addVertex(v)
                    vertices.append(i)
                new.addFace(*vertices)
        return new

    def zSlice(self, z: _Decimal):
        "slice at z"
        new = Model()
        for face in self.faces:
            if face.crossesZ(z):
                vertices = []
                for v in face.vertices:
                    i = new.addVertex(v)
                    vertices.append(i)
                new.addFace(*vertices)
        return new

    @property
    def zRange(self):
        "range for z"
        zs = [v.z for v in self.vertices]
        return (min(zs), max(zs))

def XY(x, y, r=0, quant="10000.000"):
    "Make a 2d point"
    return XYZ(x, y, r=r, quant=quant)

def parse(filename):
    "parse a .obj file"
    f = open(filename, "r")
    obj = Model()
    fonce = True
    for line in f.readlines():
        if line[0] == '#':
            continue
        if line.startswith("mtllib "):
            continue
        if line.startswith("g "):
            continue
        if line.startswith("usemtl "):
            continue
        if line.startswith("vt "):
            continue
        if line.startswith("vp "):
            continue

        if line.startswith("v "):
            xyz = [x.strip() for x in line.split(' ')[1:]]
            dxyz = [Decimal(v) for v in xyz]
            xyz = [float(v) for v in xyz]
            v = obj.addVertex(dxyz)
            print("\r%s" % (" " * 79), end='')
            print("\r%d <= Adding %s" % (v.index, tuple(xyz)), end='')
        elif line.startswith("f "):
            if fonce:
                print("")
                fonce = False
            points = [int(a.split("/")[0].strip()) for a in line.split(" ")[1:]]
            print("\r%s" % (" " * 79), end='')
            print("\rAdding %s" % (tuple(points),), end='')
            obj.addFace(*points)
    print("")
    return obj

__all__ = ['XY', 'XYZ', 'Index', 'Vertex', 'Face', 'Model', 'parse']

def main():
    "main"
    a = XY(2, 2)
    b = XY(1, 1)

    if a - b != b:
        raise ValueError("%s - %s != %s == %s" % (a, b, b, a-b))

    if a + b != XY(3, 3):
        raise ValueError

    del a, b

    if XY(1, 2) != XY(1, 2):
        raise ValueError

    if XY(0, 0) == XY(1, 1):
        raise ValueError
    if XY(0, 0) == XY(-1, -1):
        raise ValueError
    if XY(1, 0) == XY(1, -1):
        raise ValueError
    if XY(1, 0) == XY(1, 1):
        raise ValueError
    if XY(1, 0) == XY(0, 0):
        raise ValueError
    if XY(1, 0) == XY(2, 0):
        raise ValueError

if __name__ == '__main__':
    main()
