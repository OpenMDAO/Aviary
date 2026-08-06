import numpy as np
import numpy.random as rand
import random


def getVValues(x):
    """a function to get a list of non-repeating velocity values in the x data

    Args:
        x (np.ndarray): x data

    Returns:
        np.array: non-repeating velocity values
    """

    velocityValues = np.array([])  # Initializing array
    for row in x:  # Looping through rows of x-data [D, p, n, V]
        if (
            not row[3] in velocityValues
        ):  # If velocity value of current data point is not in array, append
            velocityValues = np.append(velocityValues, row[3])
    return velocityValues


# Using LHS with weights distribution using cosine
def SampleDataLHScosine(x, ct, cp):
    """A function that samples data to be used for training
        -Uses LHS, divisions distributed with |cosine| weights along velocity values
        with seeded randomness to pull one value from each division
        -Also constructs validation data from leftover data points

    Args:
        xt0 (ndarray): full x-data set [D, p, n, V]
        ct0 ([type]): full thrust coefficient set
        cp0 ([type]): full power coefficient set

    Returns:
        Training data and validation data
    """

    xt = np.ndarray((0, 4))  # Initializing arrays
    ctc = np.ndarray((0, 1))
    cpc = np.ndarray((0, 1))
    random.seed(6)  # Random seed
    rand.seed(6)  # NumPy seed
    n = 2200 # Number of division (Best chosen through trial and error)

    # Creating  a cosine weight matrix
    A = np.array([abs(np.cos(i * np.pi / (n))) for i in range(n)])
    # A=np.array([100*(0.5)**(5*i/n) for i in range(n)])

    # Using the cosine weight matrix as the weight distribution in selecting segment sizes
    segmentsize = np.array([x[:, 3].max() * (i / n) for i in range(n)])
    number_list = random.choices(segmentsize, weights=A, k=n)
    number_list.sort()

    # Looping through all divisions
    for i in range(n - 1):
        # potential values in or above current segment
        potentialValues = x[
            x[:, 3] >= number_list[i]
        ]  # Note xt0[:, 3] is extracting only velocity
        potentialT = ct[x[:, 3] >= number_list[i]]
        potentialP = cp[x[:, 3] >= number_list[i]]

        # # Restrict potential values to be less than start of next segment
        potentialT = potentialT[potentialValues[:, 3] < number_list[i + 1]]
        potentialP = potentialP[potentialValues[:, 3] < number_list[i + 1]]
        potentialValues = potentialValues[potentialValues[:, 3] < number_list[i + 1]]

        numValues = potentialValues.shape[0]

        # If there are actually values in this segment, we append a random index to the returned ndarrays
        if numValues > 0:
            randomIndex = np.random.randint(0, numValues)

            xt = np.vstack([xt, potentialValues[randomIndex]])
            ctc = np.vstack([ctc, potentialT[randomIndex]])
            cpc = np.vstack([cpc, potentialP[randomIndex]])

    xVal = np.ndarray((0, 4))  # Initializing arrays
    TVal = np.ndarray((0, 1))
    PVal = np.ndarray((0, 1))
    for row, Trow in zip(x, ct):
        if not row.tolist() in xt.tolist():
            xVal = np.vstack([xVal, row])
            TVal = np.vstack([TVal, Trow])
    for row, Prow in zip(x, cp):
        if not row.tolist() in xt.tolist():
            PVal = np.vstack([PVal, Prow])
    return xt, ctc, cpc, xVal, TVal, PVal


# The Original LHS sampler
def SampleDataLHS(xt0, ct0, cp0):
    """A function that samples data to be used for training
        -Uses LHS, even divisions along velocity values with seeded randomness to
        pull one value from each division
        -Also constructs validation data from leftover data points

    Args:
        xt0 (ndarray): full x-data set [D, p, n, V]
        ct0 ([type]): full thrust coefficient set
        cp0 ([type]): full power coefficient set

    Returns:
        Training data and validation data
    """

    xt = np.ndarray((0, 4))  # Initializing arrays
    ct = np.ndarray((0, 1))
    cp = np.ndarray((0, 1))
    rand.seed(3)  # Seed chosen based on experimentation

    vVal = getVValues(xt0)  # Getting list of non-repeating velocities
    n = 1100  # number of divisions
    segmentSize = vVal.max() / n  # Computing size of velocity division

    # Looping through all divisions
    # i * segmentSize is starting velocity value of i-th segment
    for i in range(n):
        # potential values in or above current segment
        potentialValues = xt0[
            xt0[:, 3] >= (i * segmentSize)
        ]  # Note xt0[:, 3] is extracting only velocity
        potentialT = ct0[xt0[:, 3] >= (i * segmentSize)]
        potentialP = cp0[xt0[:, 3] >= (i * segmentSize)]

        # have to check if we are in the last segment
        if i == n - 1:
            potentialT = potentialT[potentialValues[:, 3] <= vVal.max()]
            potentialP = potentialP[potentialValues[:, 3] <= vVal.max()]
            potentialValues = potentialValues[potentialValues[:, 3] <= vVal.max()]

        # Restrict potential values to be less than start of next velocity segment
        else:
            potentialT = potentialT[potentialValues[:, 3] < ((i + 1) * segmentSize)]
            potentialP = potentialP[potentialValues[:, 3] < ((i + 1) * segmentSize)]
            potentialValues = potentialValues[
                potentialValues[:, 3] < ((i + 1) * segmentSize)
            ]

        numValues = potentialValues.shape[0]

        # If there are actually values in this segment, we append a random index to the returned ndarrays
        if numValues > 0:
            randomIndex = rand.randint(0, numValues)

            xt = np.vstack([xt, potentialValues[randomIndex]])
            ct = np.vstack([ct, potentialT[randomIndex]])
            cp = np.vstack([cp, potentialP[randomIndex]])

    # Constructing validation data--the data that is left over from sampling
    xVal = np.ndarray((0, 4))
    TVal = np.ndarray((0, 1))
    PVal = np.ndarray((0, 1))

    # Looping through all data, appending to validation data if not contained within training data
    for row, Trow in zip(xt0, ct0):
        if not row.tolist() in xt.tolist():
            xVal = np.vstack([xVal, row])
            TVal = np.vstack([TVal, Trow])
    for row, Prow in zip(xt0, cp0):
        if not row.tolist() in xt.tolist():
            PVal = np.vstack([PVal, Prow])

    return xt, ct, cp, xVal, TVal, PVal
