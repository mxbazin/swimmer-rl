import numpy as np

def wrap_angle(angle): 
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return angle 

def wrap_displacement(d): 
    return (np.mod(d + np.pi, 2*np.pi) - np.pi )

# print(wrap_angle(0))          # attendu : 0
# print(wrap_angle(np.pi/2))    # attendu : 1.5708
# print(wrap_angle(3*np.pi))    # attendu : ?
# print(wrap_angle(-3*np.pi))   # attendu : ?
# print(wrap_angle(6))          # attendu : ?  (6 rad, c'est un peu moins qu'un tour)

print( wrap_displacement(0.5))     # attendu : 0.5   (pas de repliement)
print(wrap_displacement(5.0))     # attendu : 5 - 2π = -1.283
print(wrap_displacement(-5.0))    # attendu : +1.283)