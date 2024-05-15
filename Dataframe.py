import random

class DataFrame:
    """
    Dataframe class for reading and manipulating csv files. (To the likes of pandas)
    
    Parameters
    ----------------
        file: path to the csv file to be read

    """
    
    nanlist = ["?", "", " ", "NaN", "nan", "NAN", "Nan"]

    # ---------------------------------------------------------------

    # Permite criar uma estrutura de dados tabular para armazenar 
    # e manipular conjuntos de dados de forma eficiente.

    def __init__(self, file, dataframe = None, matrix = None) -> None:
        if dataframe is not None:
            self.filename = ""
            self.atributos = dataframe.atributos
            self.dados = dataframe.get_data()
            self.numEntradas = len(self.dados)
            self.targetVar = self.atributos[-1]
            self.targetCol = len(self.atributos) - 1
            self.filename = file
        elif matrix is not None:
            self.filename = ""
            self.atributos = matrix[0]
            self.dados = matrix[1:]
            self.numEntradas = len(self.dados)
            self.targetVar = self.atributos[-1]
            self.targetCol = len(self.atributos) - 1
            self.filename = file
        else:
            self.filename = ""
            self.atributos = []
            self.dados = []
            self.numEntradas = 0
            self.targetVar = ""
            self.targetCol = 0
            self.filename = file

    # ---------------------------------------------------------------

    def read_csv(self):
        with open(self.filename, "r") as csv:
            count = 0

            for line in csv:

                if count == 0:
                    line = line.strip()
                    self.atributos=line.split(',')
                    self.targetVar = self.atributos[-1]
                    self.targetCol = len(self.atributos) - 1

                else:
                    entrada=[]
                    line = line.strip()
                    entrada=line.split(',')
                    # append the data to the list
                    self.dados.append(entrada)

                count += 1
            self.numEntradas = count - 1

        # format continuous values to default ranges


    def format_continuous(self):
        # Normalizing values from Weather dataset
            if self.filename == "datasets/weather.csv":
                for i in self.dados:
                    if int(i[1]) > 80:
                        i[1] = ">80"
                    elif int(i[1]) > 75:
                        i[1] = "76-80"
                    elif int(i[1]) > 70:
                        i[1] = "71-75"
                    else:
                        i[1] = "64-70"

                    if int(i[2]) > 90:
                        i[2] = ">90"
                    elif int(i[2]) > 80:
                        i[2] = "81-90"
                    elif int(i[2]) > 70:
                        i[2] = "71-80"
                    else:
                        i[2] = "65-70"

        # Normalizing values from Iris dataset
            if self.filename == "datasets/iris.csv":
                for i in self.dados:
                    if float(i[0]) > 7.0:
                        i[0] = ">7"
                    elif float(i[0]) > 6.0:
                        i[0] = "6-7"
                    elif float(i[0]) > 5.0:
                        i[0] = "5-6"
                    else:
                        i[0] = "4-5"

                    if float(i[2]) > 4.0:
                        i[2] = ">4"
                    elif float(i[2]) > 3.5:
                        i[2] = "3.5-4"
                    elif float(i[2]) > 3.0:
                        i[2] = "3-3.5"
                    elif float(i[2]) > 2.5:
                        i[2] = "2.5-3"
                    else:
                        i[2] = "2-2.5"

                    if float(i[3]) > 6.0:
                        i[3] = ">6"
                    elif float(i[3]) > 5.0:
                        i[3] = "5-6"
                    elif float(i[3]) > 4.0:
                        i[3] = "4-5"
                    elif float(i[3]) > 3.0:
                        i[3] = "3-4"
                    elif float(i[3]) > 2.0:
                        i[3] = "2-3"
                    else:
                        i[3] = "1-2"
                    if float(i[1]) > 2.0:
                        i[1] = ">2"
                    elif float(i[1])  > 1.5:
                        i[1] = "1.5-2"
                    elif float(i[1]) > 1.0:
                        i[1] = "1-1.5"
                    elif float(i[1]) > 0.5:
                        i[1] = "0.5-1"
                    else:
                        i[1] = "0-0.5"

     # ---------------------------------------------------------------

    def get_target_col(self):
        target_col = []
        for i in self.dados:
            target_col.append(i[self.targetCol])
        return target_col
    
    # ---------------------------------------------------------------

    def get_data(self):
        data = []
        for i in self.dados:
            data.append(i)
        return data
    
    # ---------------------------------------------------------------
    
    def if_contains(self, val: str) -> list:
        list = []
        for i in self.dados:
            if val in i:
                list.append(i)
        return list
    
    def if_contains_in_column(self, val: str, col: int) -> list:
        list = []
        for i in self.dados:
            if val in i[col]:
                list.append(i)
        return list
    
    # ---------------------------------------------------------------

    def getColumn(self, string: str) -> int:
        for i in self.atributos:
            if(i == string):
                return self.atributos.index(i)

    def get_unique_values(self, col: int) -> list:
        unique_values = []
        for i in self.dados:
            if i[col] not in unique_values:
                unique_values.append(i[col])
        return unique_values

    def get_most_common_class(self):
        return max(set(self.get_target_col()), key=self.get_target_col().count)
    
    # ---------------------------------------------------------------

    def drop(self, col) -> None:

        """Remove a column from the dataframe, given its index (begins at 0)"""

        for i in range(0,len(self.dados)):
            del self.dados[i][col]

        del self.atributos[col]
        self.targetCol -= 1

    # ---------------------------------------------------------------
    