from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report




class Models:



    def logit(self, cols, target):

        if target is None:
           print("Целевая переменная не обьявлена")
           return
       
        if cols is None:
          cols = self.df.drop([target], axis=1)

        X = self.df[cols]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
    
    def rnd_forest_c(self, cols, target):

        if target is None:
           print("Целевая переменная не обьявлена")
           return
       

        if cols is None:
          cols = self.df.drop([target], axis=1)

        X = self.df[cols]
        y = self.df[target]

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
