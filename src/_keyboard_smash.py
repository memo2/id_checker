import itertools
import math

def tokenize_keyboard(board):
    return [list(row.strip()) for row in board]

def invert_grid(grid):
    out = {}
    for row_i, row in enumerate(grid):
        for col_i, cell in enumerate(row):
            if cell:
                out[cell] = (float(row_i), float(col_i))
    return out

KEYBOARD = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '='],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    ['',  '',  '',  '',  ' ',  '',  '',  '',  '',  ''],
]
KEYBOARD_GRID = invert_grid(KEYBOARD)

SHIFTED_KEYBOARD = [
    ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+'],
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?'],
    ['',  '',  '',  '',  ' ',  '',  '',  '',  '',  ''],
]
SHIFTED_KEYBOARD_GRID = invert_grid(SHIFTED_KEYBOARD)

def get_distance(a, b):
    a_pos = KEYBOARD_GRID.get(a) or SHIFTED_KEYBOARD_GRID.get(a)
    b_pos = KEYBOARD_GRID.get(b) or SHIFTED_KEYBOARD_GRID.get(b)
    if a_pos and b_pos:
        return math.hypot(a_pos[0] - b_pos[0], a_pos[1] - b_pos[1])
    return 0.0

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def score_not_mashing(text):
    "Returns a float - higher score is less likely to be mashing."
    distance = 0.0
    for a, b in pairwise(text):
        distance += (get_distance(a, b) - 1)
    return distance / len(text)

def is_mashing(text, cutoff=1.1):
    return score_not_mashing(text) < cutoff
