class SVM_Input:
    
    def __init__(self, training_data, prediciton_data, kernel, C, gamma, degree):
        self.training_data = training_data
        self.prediction_data = prediciton_data
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        
        
class SVM_Result:
    
    def __init__(self, probability, distance, yy, n_support, score):
        self.current = 0
        self.probability = probability
        self.distance = distance
        self.line = yy
        self.n_support = n_support
        self.score = score
        
        
    # Definitons to satisfy the iterable protocoll
        
    def __iter__(self):
        return self
        
    def __next__(self): 
        if self.current > 1:
            raise StopIteration
        elif self.current == 0:
            self.current += 1
            return self.probability
        else:
            self.current += 1
            return self.distance
            

class Points_Information:
    
    def __init__(self, results):
        
        # Invert the list to get the probabilites and the distances for each svm
        unzips = list(zip(*results))
        
        probabilites_for_each_svm = unzips[0]
        distances_for_each_svm = unzips[1]
        
        self.probabilites = list(zip(*probabilites_for_each_svm))
        self.distances = list(zip(*distances_for_each_svm))
        
class Bootstrap_Result:

    def __init__(self, svm, accuracy, classification, var_probability, var_distance, n_support):
        self.svm = svm
        self.accuracy = accuracy
        self.classification = classification
        self.var_probability = var_probability
        self.var_distance = var_distance
        self.n_support = n_support
        
    def view(self):
        print()
        print("Result of Bootstrap")
        print()
        print(self.svm)
        print("Accuaray:", self.accuracy)
        print("Classification:", self.classification)
        print("Variance in Probabilty:", self.var_probability)
        print("Variance in distance:", self.var_distance)
        print("Number of Suppotvectors:",self.n_support)
        
    # Definitons to satisfy the iterable protocoll
        
    def __iter__(self):
        return self
        
    def __next__(self): 
        if self.current > 5:
            raise StopIteration
        elif self.current == 0:
            self.current += 1
            return self.svm
        elif self.current == 1:
            self.current += 1
            return self.accuracy
        elif self.current == 2:
            self.current += 1
            return self.classification
        elif self.current == 3:
            self.current += 1
            return self.var_probability
        elif self.current == 4:
            self.current += 1
            return self.var_distance
        else:
            self.current += 1
            return self.n_support

    
