import math

def euclidean_distance(point1:list,point2:list)->float:
    #Calculates straight line distance betweeen two points in n-dimensional space
    return math.sqrt(sum((a-b)**2 for a,b in zip(point1,point2)))
    
def get_neighbors(training_data:list,labels:list,new_point:list,k:int)->list:
    #Calculates distances to all points and returns the labels of the k-nearest neighbors
    distances=[]
    for point,label in zip(training_data,labels):
        #Store a list of [distance,label]
        distances.append([euclidean_distance(point,new_point),label])

    #Sort by distance (index 0) in ascending order
    sorted_distances = sorted(distances,key=lambda x:x[0])

    #Slicing[:k] grabs the first k items and [1] extracts the label
    return [neighbor[1] for neighbor in sorted_distances[:k]]

def predict_classification(neighbors:list)->str:
    #Finds the most frequent label in a list of neighbors
    occurences={}
    unique = set(neighbors)
    for label in unique:
        occurences[label] = neighbors.count(label)

    #Returns the key with the highest value in the dictionary
    return max(occurences,key=occurences.get)

def normalize_dataset(dataset:list)->list:
    #Scales all features in n-dimensional dataset to a range between 0.0 and 1.0

    #zip(*dataset) unpacks the rows and zips them back as columns
    cols=list(zip(*dataset))

    #Calculate the min and max for every single column
    mins = [min(col) for col in cols]
    maxs = [max(col) for col in cols]

    normalized_data=[]
    for row in dataset:
        scaled_row=[]
        #enumerate gives both index(i) and the value (val)
        for i,val in enumerate(row):
            range_val = (maxs[i]-mins[i]) or 1 #to prevent zero division error
            scaled_val = (val-mins[i])/range_val
            scaled_row.append(scaled_val)
        normalized_data.append(scaled_row)
    return normalized_data

def knn_predict(training_data:list,labels:list,new_point:list,k:int)->str:
    #Scales all data, finds nearest neighbors, and returns a classification
    combined_data = training_data+[new_point]
    scaled_combined = normalize_dataset(combined_data)
    scaled_new_point = scaled_combined.pop()
    scaled_training_data = scaled_combined

    return predict_classification(get_neighbors(scaled_training_data,labels,scaled_new_point,k))