import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equals(self, other):
        if (self.x == other.x and self.y == other.y):
            return True
        else:
            return False

    def distance(self, other):
        delta_x = other.x - self.x
        delta_y = other.y - self.y
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)
        return distance

    def __str__(self):
        return "({},{})".format(self.x, self.y)

    def __array__(self):
        return np.array([self.x,self.y])

class guassian:
    def __init__(self, standard_deviation, center_point):
        self.deviation = standard_deviation
        self.covariance_matrix = [[standard_deviation,0],[0,standard_deviation]]
        self.mean = center_point

    def pdf(self,x_point):
        coef = 1/(2*math.pi)
        det = 1/(math.sqrt(np.linalg.det(self.covariance_matrix)))
        exponent = -1/2 * self.mean.distance(x_point) ** 2#np.transpose(np.subtract(x_point,self.mean)) * np.linalg.inv(self.covariance_matrix) * np.subtract(x_point,self.mean)
        exp = math.exp(exponent)
        return coef * det * exp

q1_sites = []
q1_sites.append(point(0,1))
q1_sites.append(point(3,2))
q1_sites.append(point(3,-2))

q1_data = []
q1_data.append(point(0,0))
q1_data.append(point(7,-1))
q1_data.append(point(7,1))
q1_data.append(point(-6,3))
q1_data.append(point(-1,3))

q1_closest = [point(0,0),point(0,0),point(0,0),point(0,0),point(0,0)]
for i in range(0,5):
    curr_point = q1_data[i]
    min_distance = 10000.0
    for pnt in q1_sites:
        curr_distance = curr_point.distance(pnt)
        if(curr_distance < min_distance):
            q1_closest[i] = pnt
            min_distance = curr_distance

for i in range(0,5):
    print('the closest point to {} is {}'.format(q1_data[i],q1_closest[i]))

q1_g1 = guassian(1.0, point(0,1))
q1_g2 = guassian(2.0, point(3,2))
q1_g3 = guassian(4.0, point(3,-2))

for i in range(0,5):
    print("Weights for {}\n".format(q1_data[i]))
    g1_pdf = q1_g1.pdf(q1_data[i])
    g2_pdf = q1_g2.pdf(q1_data[i])
    g3_pdf = q1_g3.pdf(q1_data[i])
    g_sum = g1_pdf + g2_pdf + g3_pdf
    print("{},{},{}\n".format(g1_pdf/g_sum,g2_pdf/g_sum,g3_pdf/g_sum))

q2_data = np.array([[-5,5],[-5,4],[-5,3],[-4,5],[-4,4],[-4,3],[-3,5],[-3,4],[-3,3],[5,5],[5,4],[5,3],[4,5],[4,4],[4,3],[3,5],[3,4],[3,3],[0,2],[0,1],[-1,1],[-1,-1],[0,-1],[0,0],[-1,-2],[1,-2]])
q2_kmeans = KMeans(n_clusters=2).fit(q2_data)

plot_data = np.transpose(q2_data)
centers = q2_kmeans.cluster_centers_
plt.plot(plot_data[0],plot_data[1],'ro')
plt.plot(centers[0],centers[1],'bo')
plt.show()

print(centers)
print(q2_kmeans.inertia_)

q4_data = np.array([[0,1,1],[-1,2,1],[1,2,1],[1,0,-1],[-1,0,-1],[2,2,-1],[-2,2,-1]])
q4_plot_data = np.transpose(q4_data)
q4_positive_x = []
q4_negative_x = []
q4_positive_y = []
q4_negative_y = []
for i in range(0,len(q4_plot_data[2])):
    if (q4_plot_data[2][i] == 1):
        q4_positive_x.append(q4_plot_data[0][i])
        q4_positive_y.append(q4_plot_data[1][i])
    else:
        q4_negative_x.append(q4_plot_data[0][i])
        q4_negative_y.append(q4_plot_data[1][i])

plt.clf()
plt.plot(q4_positive_x,q4_positive_y, 'bo')
plt.plot(q4_negative_x,q4_negative_y, 'ro')
plt.show()



print('done')