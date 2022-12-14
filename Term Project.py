#imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import math
import numpy as np

#reading data frame and removing unnecessary columns
df = pd.read_csv("C:\\Users\\thorn\\Desktop\\Coding\\CSVs\\space_decay.csv")
df = df.drop(['CCSDS_OMM_VERS', 'COMMENT', 'ORIGINATOR', 'CENTER_NAME', 'REF_FRAME', 'TIME_SYSTEM', 'MEAN_ELEMENT_THEORY',\
              'EPHEMERIS_TYPE', 'CLASSIFICATION_TYPE', 'ELEMENT_SET_NO', 'MEAN_MOTION_DDOT', 'DECAY_DATE'], axis = 1)

#plotting a heatmap to see what columns may be correlated
fig, ax = plt.subplots()
ax = sns.heatmap(df.corr(), vmin = -1, cmap = 'vlag')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)

#creating a new dataframe with all unique years to graph new objects and total objects
totalCounts = pd.DataFrame(df['LAUNCH_DATE'].value_counts()).sort_index()
totalCounts.reset_index(inplace = True)
totalCounts = totalCounts.rename(columns = {'index':'LAUNCH_DATE', 'LAUNCH_DATE':'NEW_ITEMS'})

#calculating total objects
totalItems = []
number = 0
for idx, row in totalCounts.iterrows():
    number = number + row['NEW_ITEMS']
    totalItems.append(number)
totalCounts['TOTAL_ITEMS'] = totalItems

#defining the curve function
def curveFunc(x, a, b, c):
    return(a*x**2 + b*x + c)

#calculating parameters for the total items graph
[a,b,c], pcov = curve_fit(curveFunc, totalCounts['LAUNCH_DATE'], totalCounts['TOTAL_ITEMS'])

#calculating predicted values for the total items graph
predsTotal = []
for value in totalCounts['LAUNCH_DATE']:
    predsTotal.append(round(a*value**2 + b*value + c, 2))
totalCounts['PREDS_TOTAL'] = predsTotal

#predicting total debris in 2022, 2023, and 2030
predTotal2022 = round(a*2022**2 + b*2022 + c)
predTotal2023 = round(a*2023**2 + b*2023 + c)
predTotal2030 = round(a*2030**2 + b*2030 + c)
print('Predicted total debris in orbit in 2022: ' + str(predTotal2022))
print('Predicted total debris in orbit in 2023: ' + str(predTotal2023))
print('Predicted total debris in orbit in 2030: ' + str(predTotal2030))

#calculating parameters for the new items graph
[d, e, f], pcov = curve_fit(curveFunc, totalCounts['LAUNCH_DATE'], totalCounts['NEW_ITEMS'])

#calculating predicted values for the new items graph
predsNew = []
for value in totalCounts['LAUNCH_DATE']:
    predsNew.append(round(d*value**2 + e*value + f, 2))
totalCounts['PREDS_NEW'] = predsNew

#predicting new debris in 2022, 2023, and 2030
predNew2022 = round(d*2022**2 + e*2022 + f)
predNew2023 = round(d*2023**2 + e*2023 + f)
predNew2030 = round(d*2030**2 + e*2030 + f)
print('Predicted new debris in orbit in 2022: ' + str(predNew2022))
print('Predicted new debris in orbit in 2023: ' + str(predNew2023))
print('Predicted new debris in orbit in 2030: ' + str(predNew2030))

#plotting total items vs year and a line of best fit
fig, ax = plt.subplots()
ax.scatter(totalCounts['LAUNCH_DATE'], totalCounts['TOTAL_ITEMS'])
ax.plot(totalCounts['LAUNCH_DATE'], totalCounts['PREDS_TOTAL'])
ax.set_title('Total amount of debris in orbit vs year')
ax.set_xlabel('Year')
ax.set_ylabel('Items of debris in orbit')

#plotting new items vs year and a line of best fit
fig, ax = plt.subplots()
ax.scatter(totalCounts['LAUNCH_DATE'], totalCounts['NEW_ITEMS'])
ax.plot(totalCounts['LAUNCH_DATE'], totalCounts['PREDS_NEW'])
ax.set_title('New debris in orbit vs year')
ax.set_xlabel('Year')
ax.set_ylabel('New items of debris in orbit')

#calculating true anomaly and adding a column
trueAnomalyList = []
for idx, row in df.iterrows():
    diff = 100.0
    torad = math.pi/180.0
    eOld = math.pi/180.0
    e = row['ECCENTRICITY']
    meanAnomaly = row['MEAN_ANOMALY']
    eOld = meanAnomaly + e / 2.0
    while(diff > 0.0001): #defines how many decimals will be on the number
        eNew = eOld - (eOld - e * (math.sin(eOld)) - meanAnomaly) / (1 - e * (math.cos(eOld)))
        diff = abs(eOld - eNew)
        eOld = eNew
    eccentricAnomaly = eNew/torad
    trueAnomaly = math.acos((math.cos(eccentricAnomaly) - e) / (1 - e * math.cos(eccentricAnomaly)))
    trueAnomaly = trueAnomaly / torad
    trueAnomalyList.append(trueAnomaly)
df['TRUE_ANOMALY'] = trueAnomalyList

#calculating angular momentum and adding a column
angularMomentumList = []
for idx, row in df.iterrows():
    angularMomentum = row['SEMIMAJOR_AXIS'] * (1 - row['ECCENTRICITY']**2)
    angularMomentumList.append(angularMomentum)
df['ANGULAR_MOMENTUM'] = angularMomentumList

#creating lists for the x, y, z positions and x, y, z velocities of each piece of debris
xPoss = []
yPoss = []
zPoss = []
xVels = []
yVels = []
zVels = []
#earth's gravitational constant
mu = 3.986004418E+05
#earth's radius
r_Earth = 6378.14
#using given and calculated information to find the x, y, z positions and velocities of each piece of debris
for idx, row in df.iterrows():
    #calculating the initial positional state vector
    posStateVector = (row['ANGULAR_MOMENTUM']**2 / mu) * (1 / (1 + row['ECCENTRICITY'] * \
                                                               math.cos(math.radians(row['TRUE_ANOMALY']))))\
        * np.array([math.cos(math.radians(row['TRUE_ANOMALY'])), math.sin(math.radians(row['TRUE_ANOMALY'])), 0])
    #calculating the initial velocity state vector
    velStateVector = (mu / row['ANGULAR_MOMENTUM']) * np.array([-math.sin(math.radians(row['TRUE_ANOMALY'])),\
                                                                (row['ECCENTRICITY'] + \
                                                                 math.cos(math.radians(row['TRUE_ANOMALY']))), 0])
    #defining the inverted direction cosine matrix
    dirCosMatrixInv = np.array([[math.cos(math.radians(-1 * row['ARG_OF_PERICENTER'])), \
                      math.sin(math.radians(-1 * row['ARG_OF_PERICENTER'])), 0],\
                      [-1 * math.sin(math.radians(-1 * row['ARG_OF_PERICENTER'])),\
                      math.cos(math.radians(-1 * row['ARG_OF_PERICENTER'])), 0],\
                      [0, 0, 1]]) @ np.array([[1, 0, 0],\
                      [0, math.cos(math.radians(-1 * row['INCLINATION'])),\
                      math.sin(math.radians(-1 * row['INCLINATION']))],\
                      [0, -1 * math.sin(math.radians(-1 * row['INCLINATION'])),\
                      math.cos(math.radians(-1 * row['INCLINATION']))]]) @ \
                      np.array([[math.cos(math.radians(-1 * row['RA_OF_ASC_NODE'])),\
                      math.sin(math.radians(-1 * row['RA_OF_ASC_NODE'])), 0],\
                      [-1 * math.sin(math.radians(-1 * row['RA_OF_ASC_NODE'])),\
                      math.cos(math.radians(-1 * row['RA_OF_ASC_NODE'])), 0],\
                      [0, 0, 1]])
    #multiplying the cosine matrix with the position state vector to get the equatorial position state vector
    eqPosVector = dirCosMatrixInv @ posStateVector
    #multiplying the cosine matrix with the velocity state vector to get the equatorial velocity state vector
    eqVelVector = dirCosMatrixInv @ velStateVector
    xPoss.append(eqPosVector[0])
    yPoss.append(eqPosVector[1])
    zPoss.append(eqPosVector[2])
    xVels.append(eqVelVector[0])
    yVels.append(eqVelVector[1])
    zVels.append(eqVelVector[2])
#appending new columns
df['X_POS'] = xPoss
df['Y_POS'] = yPoss
df['Z_POS'] = zPoss
df['X_VEL'] = xVels
df['Y_VEL'] = yVels
df['Z_VEL'] = zVels

#creating a function that models the orbit
def modelOrbit(state, t):
    mu = 3.986004418E+05
    x = state[0]
    y = state[1]
    z = state[2]
    xDot = state[3]
    yDot = state[4]
    zDot = state[5]
    xDdot = -mu * x / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    yDdot = -mu * y / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    zDdot = -mu * z / (x ** 2 + y ** 2 + z ** 2) ** (3 / 2)
    dstate_dt = [xDot, yDot, zDot, xDdot, yDdot, zDdot]
    return(dstate_dt)

#simulating the passage of time
t = np.linspace(0, 6*1200, 200)

#setting constants
N = 50
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)
theta, phi = np.meshgrid(theta, phi)

#setting earth's parameters to scale
r_Earth = 63.7814
X_Earth = r_Earth * np.cos(phi) * np.sin(theta)
Y_Earth = r_Earth * np.sin(phi) * np.sin(theta)
Z_Earth = r_Earth * np.cos(theta)

xs = []
ys = []
zs = []
indexes = [69,68,73] #change these numbers to plot different pieces of debris (0 - 14371)
for item in indexes:
    objState = df.loc[item, ['X_POS', 'Y_POS', 'Z_POS', 'X_VEL', 'Y_VEL', 'Z_VEL']]
    
    #calculating the state of the debris
    sol = odeint(modelOrbit, objState, t)
    xSat = sol[:, 0]
    xs.append(xSat)
    ySat = sol[:, 1]
    ys.append(ySat)
    zSat = sol[:, 2]
    zs.append(zSat)
    
#graphing the orbit
titleList = [x + 1 for x in indexes]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
i = 0
for item in xs:
    ax.plot3D(xs[i], ys[i], zs[i], 'black')
    i = i + 1
ax.view_init(30, 135)
ax.set_aspect('equal')
plt.title('Plot of space debris and earth \nDebris item(s): ' + str(titleList))
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')