"""Wire format used in the 2025 team integration: one 'id x y yaw' per line."""
from math import isfinite

def parse_states(message: bytes):
    result = {}
    for lineno, line in enumerate(message.decode('utf-8').splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise ValueError(f'line {lineno}: expected id x y yaw')
        vid = int(fields[0]); x,y,yaw = map(float, fields[1:])
        if vid < 0 or not all(map(isfinite,(x,y,yaw))):
            raise ValueError(f'line {lineno}: invalid vehicle state')
        result[vid] = (x,y,yaw)
    return result

def encode_states(states):
    return ('\n'.join(f'{int(i)} {x:.6f} {y:.6f} {yaw:.6f}' for i,(x,y,yaw) in states.items())+'\n').encode('utf-8')
