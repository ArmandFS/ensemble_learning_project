import numpy as np
import dtree

class randomized_majority_voter:
    """
    modified majority voter class with randomization
    """
    def __init__(self):
        self.tree = dtree.dtree()

    def make_classifiers(self, data, targets, features, which_gain, nTrees, nSamples, nFeatures, maxlevel=5):
        nPoints = np.shape(data)[0]
        self.nSamples = nSamples
        self.nTrees = nTrees
        classifiers = []
        #this will all initialize weights to 1
        self.weights = np.ones(nTrees)  

        for i in range(nTrees):
            samplePoints = np.random.randint(0, nPoints, (nPoints, nSamples))
            sample, sampleTarget = [], []

            for j in range(nSamples):
                for k in range(nPoints):
                    sample.append(data[samplePoints[k, j]])
                    sampleTarget.append(targets[samplePoints[k, j]])
            
            classifiers.append(
                self.tree.make_tree(sample, sampleTarget, features, which_gain, maxlevel, forest=nFeatures)
            )
        
        return classifiers

    def randomized_vote(self, classifiers, data, targets, beta=0.6):
        decision = []
        
        for j in range(len(data)):
            outputs = []
            classifier_indices = []
            
            for i in range(self.nTrees):
                out = self.tree.classify(classifiers[i], data[j])
                if out is not None:
                    outputs.append(out)
                    classifier_indices.append(i)  
            
            if len(outputs) == 0:
                decision.append(None)
                continue
            
            unique_outputs = list(set(outputs))
            frequency = np.zeros(len(unique_outputs))
            mass = np.zeros(len(unique_outputs))
            
            outputs = np.array(outputs)
            
            for idx, label in enumerate(unique_outputs):
                inds = np.where(outputs == label)[0] 
                frequency[idx] = len(inds) 
                #sum all the classified weights
                mass[idx] = np.sum(self.weights[np.array(classifier_indices)[inds]])  
            
            #Normalize mass to create probabilities
            if np.sum(mass) > 0:
                mass /= np.sum(mass)  
            else:
                mass = np.ones(len(unique_outputs)) / len(unique_outputs)  
            
            #Randomly choose a classifier based on mass probabilities
            chosen_label_idx = np.random.choice(len(unique_outputs), p=mass)
            final_decision = unique_outputs[chosen_label_idx]
            decision.append(final_decision)
            
            if final_decision != targets[j]:
                incorrect_classifiers = np.where(outputs == final_decision)[0]
                self.weights[np.array(classifier_indices)[incorrect_classifiers]] *= beta  
            else:
                correct_classifiers = np.where(outputs == final_decision)[0]
                self.weights[np.array(classifier_indices)[correct_classifiers]] *= (1 / beta)  
        
        print("Updated weights:", self.weights)
        return decision
