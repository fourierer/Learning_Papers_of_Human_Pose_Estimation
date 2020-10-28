import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()
polygons = []
num_sides = 100
gemfield_polygons = [[246.9,404.24,261.04,378.46,269.35,316.1,269.35,297.8,290.98,252.07,299.29,227.12,
                    302.62,173.07,286.81,143.97,285.16,133.16,310.1,124.01,321.74,116.52,335.05,117.35,
                    351.68,119.02,359.99,128.16,348.35,137.31,341.7,152.28,361.66,163.93,384.94,190.53,
                    401.57,208.82,394.09,219.64,386.6,221.3,375.79,214.65,370.8,201.34,366.64,256.22,
                    360.83,270.36,369.14,284.5,367.47,320.25,383.28,350.19,401.57,386.78,409.05,408.4,
                    408.22,420.87,400.74,431.68,389.1,425.03,381.61,395.93,370.8,366.82,343.37,321.92,
                    329.22,288.65,310.1,313.6,299.29,312.77,285.98,332.73,283.49,362.66,276.0,392.6,
                    281.83,399.25,259.37,416.71,236.92,416.71,236.92,407.56]]
gemfield_polygon = gemfield_polygons[0]
max_value = max(gemfield_polygon) * 1.3
gemfield_polygon = [i * 1.0/max_value for i in gemfield_polygon]
poly = np.array(gemfield_polygon).reshape((int(len(gemfield_polygon)/2), 2))
polygons.append(Polygon(poly,True))
p = PatchCollection(polygons, cmap=matplotlib.cm.jet, alpha=0.4)
colors = 100*np.random.rand(1)
p.set_array(np.array(colors))

ax.add_collection(p)

plt.savefig("polygon.png")
plt.show()



