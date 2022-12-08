import numpy as np
import pandas as pd
import csv
def chordsToArray(chords):
    chords = chords.replace('[',"")
    chords = chords.replace(']',"")
    chords = chords.replace(' ',"")
    chords = chords.split(",")
    chords = np.asarray([int(i) for i in chords])
    return chords

def keySigtoNum(keySig):
    for i in range(0,keySig.size):
        keySig[i] = keySigDict[keySig[i]]
    return keySig

def findDistance(song, cluster):
    diff = []
    for i in range(0,song.size):
        diff.append(np.power((song[i] - cluster[i]),2))
    diff = np.array(diff)
    size = diff.size
    diff = np.sum(diff)/size
    return diff

#helper functions for kmeans
def assignClusters(data, clusters):
    assignments = []
    for song in data:
        curDiffs = []
        for c in clusters:
            curDiffs.append(findDistance(song, c))
        curDiffs = np.array(curDiffs)
        curAssignment = np.where(curDiffs == np.max(curDiffs))[0][0]
        assignments.append(curAssignment)

    return assignments

def createNewClusters(data, clusterAssignment,numClusters):
    clusterDict = {}
    clustersFound = []
    dataSize = data.shape[1]
    newClusterArray = np.zeros(shape=(numClusters, data.shape[1]))
    for i in range(0,len(clusterAssignment)):
        assignment = clusterAssignment[i]
        if assignment in clustersFound:
            curArray = clusterDict[assignment]
            curArray = np.vstack((curArray, data[i]))
            clusterDict[assignment] = curArray
        else:
            clustersFound.append(assignment)


            clusterDict[assignment] = data[i]

    for cluster in clusterDict:
        curData = clusterDict[cluster]
        newCluster = []
        for i in range(0,dataSize):
            try:
                newVal = np.sum(curData[:,i])/curData[:,i].size
                newCluster.append(newVal)
            except:
                newCluster = curData
                break
            

        newClusterArray[cluster] = newCluster
    return newClusterArray    

def printResult(clusterAssignment,labels):
    clusterDict = {}
    clustersFound = []
    for i in range(0,len(clusterAssignment)):
        assignment = clusterAssignment[i]
        if assignment in clustersFound:
            curArray = clusterDict[assignment]
            curArray = curArray.append(labels[i])
            clusterDict[assignment] = curArray
        else:
            clustersFound.append(assignment)
            clusterDict[assignment] = [labels[i]]
    print(clusterDict)

#Note found "-" after some of the key signatures changes are marked but will be removed temporarily
df = pd.read_csv('inputs_test.csv')
first = True
keySigDict = {"A major": 3, "B major":5, "C major":0, "D major":2, "E major":4, "F major": -1, "G major":1, "a minor": 0, "b minor":2, "c minor":-3, "d minor":-1, "e minor":1, "f minor": -4, "g minor":-2}
df = df.values
labels = df[:,4]
data = df[:,1]
data = np.true_divide(data, np.max(data))
chords = df[:,3]
keySigs = df[:,2]
keySigs = keySigtoNum(keySigs)
keySigs = np.true_divide(keySigs, 5)
data = np.column_stack((data, keySigs))


count = 0
first = True
maxChord = 0
for row in chords:
    array = chordsToArray(row)
    array = np.true_divide(array, np.max(array))

    if first:
        first = False
        chordArray = np.array(array)
        continue
    chordArray = np.vstack((chordArray, array))

data = np.append(data, chordArray,1)
#for output file
data = np.column_stack((data, labels))
dataSize = data[:,0].size
columns = []
for i in range(data.shape[1]):
    if i == 0:
        columns.append('bpm')
    if i == 1:
        columns.append('keySig')
    if i == (data.shape[1]-1):
        columns.append('song_name')
        continue
    if i > 1:
        columns.append('chord' + str(i))
output = pd.DataFrame(data, columns=columns)
output.to_csv('normalizedTestFile.csv')
#time to make k means clustering algorithm
numClusters = 3 #we'll have to test different values might depend on size of input
# clusters = np.random.random_sample(size=(data.shape[0],numClusters))
# clusterAssignments = []
# newClusterAssignments = np.random.randint(0,numClusters, size=data.shape[0])
# while  not np.array_equal(clusterAssignments,newClusterAssignments):
     
#     clusterAssignments = newClusterAssignments
#     clusters = createNewClusters(data, clusterAssignments, numClusters)
#     newClusterAssignments = assignClusters(data, clusters)
#     print(newClusterAssignments)
#     print(clusters)
# printResult( clusterAssignments, labels)
    
    
