import numpy as np #for mathematical calculation
import sys
import copy

from Dataframe import *

def calc_total_entropy(train_data: DataFrame, class_list: list) -> float:
    '''
    ### Calculates the total entropy of the dataset

    ---

    Parameters:

    - train_data : DataFrame of the dataset
    '''

    total_row = train_data.numEntradas #the total size of the dataset
    total_entr = 0 # total entropy of the dataset
    
    for c in class_list: #for each class in the label

        total_class_count = 0 # number of rows with that class
        total_class_count = len(train_data.if_contains(c))

        total_class_entr = 0
        if total_class_count != 0:
            total_class_entr = (float)(- (total_class_count/total_row)*np.log2(total_class_count/total_row)) #entropy of the class
        else:
            total_class_entr = 0 # if there are no rows with that class, entropy is 0 (log(0) is undefined)

        total_entr += total_class_entr #adding the class entropy to the total entropy of the dataset

    return total_entr

def calc_entropy(feature_value_data: DataFrame, class_list: list) -> float:

    '''
    
    A entropia é a medida de grau de incerteza de um conjunto de dados. Isto se refere
    a quantificação da distribuição estatística deste conjunto de dados em classes, tendo em
    conta a falta de homogeneidade. Quanto maior a entropia, maior a incerteza.
    
    Portanto,
    quanto maior a diferença de entropia, menor o ganho de informação; e quanto menor a
    diferença entropia, maior o ganho de informação, indicando um atributo mais útil para a
    classificação.
    
    ### Calculates the entropy of the feature value

    Parameters
    ---

    - feature_value_data : DataFrame of the dataset with only the rows with that feature value
    - class_list : list of the classes in the label
    '''

    class_count = feature_value_data.numEntradas
    entropy = 0
    
    for c in class_list:
        label_class_count = len(feature_value_data.if_contains(c)) #row count of class c 
        
        entropy_class = 0

        if label_class_count != 0:
            probability_class = label_class_count/class_count #probability of the class
            entropy_class = - probability_class * np.log2(probability_class)  # entropy
        entropy += entropy_class # adding the entropy of the class to the total entropy of the feature value

    return entropy

def calc_info_gain(feature_name, train_data: DataFrame, class_list: list):
    '''
    ### Calculates the information gain of the feature

    Parameters

    ---

    - feature_name : name of the feature
    - train_data : DataFrame of the dataset
    - class_list : list of the classes in the label
    '''
    feature_value_list = train_data.get_unique_values(train_data.getColumn(feature_name)) #unqiue values of the feature

    total_row = train_data.numEntradas
    feature_info = 0.0
    
    for feature_value in feature_value_list:
        feature_value_data = train_data.if_contains(feature_value) #filtering rows with that feature_value
        feature_value_data.insert(0, train_data.atributos)
        feature_value_data = DataFrame("nan", matrix= feature_value_data)

        feature_value_count = feature_value_data.numEntradas
        feature_value_entropy = calc_entropy(feature_value_data, class_list) #calculcating entropy for the feature value
        feature_value_probability = feature_value_count/total_row
        feature_info += feature_value_probability * feature_value_entropy #calculating information of the feature value

        
    return calc_total_entropy(train_data, class_list) - feature_info #calculating information gain by subtracting


def find_most_informative_feature(train_data: DataFrame, class_list: list):
    '''
    ### Finds the most informative feature in the dataset

    --- 

    Parameters

    - train_data : DataFrame of the dataset
    - class_list : list of the classes in the label
    '''

    feature_list = train_data.atributos #finding the feature names in the dataset
    max_info_gain = -1
    max_info_feature = None
    
    for feature in feature_list[:-1]:  #for each feature in the dataset
        feature_info_gain = calc_info_gain(feature, train_data, class_list)

        if max_info_gain < feature_info_gain: #selecting feature name with highest information gain
            max_info_gain = feature_info_gain
            max_info_feature = feature
    
    return max_info_feature


def make_tree(root: dict, train_data: DataFrame, label: str, class_list: list):
    '''
    ### Makes the decision tree
    This function is called recursively to make the decision tree
    where it makes subtrees based on the values of the most informative feature
    and calls itself for each subtree, unless the dataset becomes empty, 
    if there are no more features or if all the rows have same class.

    ---

    Parameters

    - root : root of the tree (or subtree)
    - train_data : DataFrame of the dataset
    - label : name of the label (target feature)
    - class_list : list of the classes in the label
    '''
    if len(train_data.get_data()) != 0: #if dataset becomes enpty after updating

        # if all the rows have same class, return the class
        if len(train_data.get_unique_values(train_data.getColumn(label))) == 1:
            root = train_data.get_unique_values(train_data.getColumn(label))[0]
            return root

        # if there are no more features, return the most common class
        if len(train_data.atributos) == 1:
            
            root = train_data.get_most_common_class()
            return root

        # if there are more features, find the most informative feature
        max_info_feature = find_most_informative_feature(train_data, class_list) #most informative feature    

        # separate the dataset based on the most informative feature and make subtrees based on the values of the feature
        feature_value_list = train_data.get_unique_values(train_data.getColumn(max_info_feature)) #unqiue values of the feature

        tree= {} #root of the tree

        col = train_data.getColumn(max_info_feature)

        for feature_value in feature_value_list:

            # the tree is actually built in this format:
            # {feature_name: [feature_value, subdata, class_count, subtree]}
            # where the subtree is a dictionary with the same format

            #this way, we can easily traverse the tree and find the class of a row, the entry count of a class, etc.
            node = [] 
            node.append(max_info_feature)

            # reset subdata
            sub_data = []

            # make deepcopies before passing to the function to avoid changing the original dataset (python am i right?)
            sub_data = copy.deepcopy(train_data.if_contains_in_column(feature_value, col)) #filtering rows with that feature_value
            atributos = copy.deepcopy(train_data.atributos)

            # add the attribute names to the subdata
            sub_data.insert(0, atributos)

            sub_data = DataFrame("nan", matrix= sub_data) # convert to dataframe

            sub_data.drop(col) #dropping the feature column

            node.append(sub_data.numEntradas)

            # (this is a base case inside the recursive function, how curious)
            # if there are no more entries, before we make a subtree, we find the most common class and return it
            if len(sub_data.get_data()) == 0:

                # define the leaf node as the most common class
                root[max_info_feature][feature_value] = train_data.get_most_common_class()
                return tree
            
            tree.update({feature_value: {}}) #updating the tree with the feature name

            # recursive call to make_tree
            sub_dict = make_tree(tree[feature_value], sub_data, label, class_list)
            # add the subtree to the node
            node.append(sub_dict)
            tree[feature_value] = node #updating the tree with the subtree

    return tree

def print_dictionary(dictionary, indent=''):

    # this function is used to print the tree in a readable format as it has a lot of nested dictionaries with lists
    for key, value in dictionary.items():
        attribute, count, sub_dict = value

        print(indent + attribute + ":")
        print(indent + '    ' + key + ':' + " (" + str(count) + ")", end= '')
        
        if isinstance(sub_dict, str):
            print(" " + sub_dict)
        else:
            print("")
            print_dictionary(sub_dict, indent + "    ")

# lero print

def format_input(input: str, df: DataFrame, f: list) -> DataFrame:
    '''
    ### Formats the input string
    This function is used to format the input string to a list of strings
    where each string is a feature value

    ---

    Parameters

    - input : input string
    '''
    input = input.split(",")
    row = [i.strip() for i in input]
    list = []
    list.append(f)
    list.append(row)
    df = DataFrame("datasets/" + sys.argv[1] + ".csv", matrix= list)

    df.format_continuous() 
    return df
    

def id3(train_data: DataFrame, TarCol: int) -> dict:
    tree = {} #tree which will be updated
    class_list = train_data.get_unique_values(TarCol) #getting unqiue classes of the label

    label = train_data.atributos[TarCol] #getting the label name
    return make_tree(tree, train_data, label, class_list) #start calling recursion


def predict_target(dictionary, feat: DataFrame) -> str:
    '''
    ### Predicts the target class
    This function is used to predict the target class of a row
    by traversing the tree

    ---

    Parameters

    - dictionary : the decision tree
    - feat : the feat of the row
    '''


    if isinstance(dictionary, str):
        return dictionary

    # get the first key of the dictionary
    key = list(dictionary.keys())

    attribute = dictionary[key[0]][0]

    # get the index of the feature in the row
    index = feat.getColumn(attribute)

    for key, value in dictionary.items():
        if isinstance(value[2], str) and key == feat.get_data()[0][index]:
            return value[2]

        if key == feat.get_data()[0][index]:
            sub_dict = value[2]
            return predict_target(sub_dict, feat)
        
    return "No class found"


def predict_file(filepath: str, tree: dict, df: DataFrame) -> None:
    '''
    ### Predicts the target class of a file
    This function is used to predict the target class of a file
    by traversing the tree

    ---

    Parameters

    - filepath : the filepath of the file
    '''
    file = open(filepath, "r")
    lines = file.readlines()
    lines.pop(0)
    file.close()

    f = copy.copy(df.atributos)

    for line in lines:
        print("Input: " + line, end="")
        df = format_input(line, df, f)
        print("Predicted class: " + predict_target(tree, df) + "\n")


# ------------------ main ------------------
def main():
    df = DataFrame("datasets/" + sys.argv[1] + ".csv") #importing the dataset from the disk

    df.read_csv() #reading the dataset
    df.drop(0) # drop the ID row
    df.format_continuous() # format continuous data (only works for iris and weather)

    tree = id3(df, df.targetCol)

    print_dictionary(tree)


    # ------------------ testing ------------------

    print("\nWant to test prediction? (y/n)")
    if input() == "n":
        return

    print("Input filepath: ", end="")
    filepath = input()
    print("")
    predict_file(filepath, tree, df)


if __name__ == "__main__":
    main()