from collections import namedtuple


TGravity = namedtuple("Gravity", ["x", "y", "z"])
TPosition = namedtuple("Position", ["x", "y", "z"])
TOrientation = namedtuple("Orientation", ["roll", "pitch", "yaw"])
TPose = namedtuple("Position", ["x", "y", "z", "roll", "pitch", "yaw"])
