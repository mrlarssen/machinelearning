import pandas as pd
import numpy as np

#np.set_printoptions(threshold='nan', suppress=True)

class ChurnData:
    
    def load_data(self):

        df_main = pd.read_csv('churn.csv')
        
        # Isolate target data
        churn_result = df_main['Churn?']
        df_churn_result = pd.DataFrame(churn_result)
        
        # Convert Boolean values to Binary values
        Y_ = np.where(churn_result == 'True.', 1, 0)

        Y = []
        
        one_hot_dict = {
            0: [1,0],
            1: [0,1]
        }
        
        for i,v in enumerate(Y_):
            Y.append(one_hot_dict[v])
            
        # Dropping unneccessary columns and creating our feature space
        to_drop = ['State','Area Code','Phone','Churn?']
        df_churn_feature_space = df_main.drop(to_drop,axis=1)
        
        # Convert yes/no to Boolean values
        yes_no_cols = ["Int'l Plan", "VMail Plan"]
        df_churn_feature_space[yes_no_cols] = df_churn_feature_space[yes_no_cols] == 'yes'
        
        #Feature space as matrix
        X = df_churn_feature_space.as_matrix().astype(np.float)
        
        #Normalizing each feature in X within a range of -1.0 to 1.0
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(np.array(Y).shape)
        print(np.array(Y_).shape)

        return X,Y
        
        
        
ChurnData().load_data()
        
        
        
        