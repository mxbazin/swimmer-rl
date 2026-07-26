import numpy as np

def wrap_angle(angle): 
    angle = np.arctan2(np.sin(angle), np.cos(angle))
    return angle 

print(wrap_angle(0))          # attendu : 0
print(wrap_angle(np.pi/2))    # attendu : 1.5708
print(wrap_angle(3*np.pi))    # attendu : ?
print(wrap_angle(-3*np.pi))   # attendu : ?
print(wrap_angle(6))          # attendu : ?  (6 rad, c'est un peu moins qu'un tour)
