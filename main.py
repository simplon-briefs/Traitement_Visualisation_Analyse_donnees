# 1.Déploiement de l’environnement
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm 
from statsmodels.formula.api import ols

# 2.Sources de données : Manipulation
credits_immo_csv = pd.read_csv("Data_Vis/credit_immo.csv")
credits_immo_json = pd.read_json("Data_Vis/credit_immo.json")
credits_immo_xls = pd.read_excel("Data_Vis/credit_immo.xls")

ar = np.array([[1.1, 2, np.nan, 4], [2.7, np.nan, 5.4, 7], [5.3, 9, 1.5, 15]])
df = pd.DataFrame(ar, index = ['a1', 'a2', 'a3'], columns = ["taux_de_ventes", "croissance_vente", "ratio_benefice", "ratio_perte"])
print(df)


# 3.Traitement des données
#3.2
df = pd.read_csv("Data_Vis/credit_immo.csv")
#3.3
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
num_col = credits_immo_csv.select_dtypes(include="number").columns
im = SimpleImputer()
imputed = im.fit_transform(credits_immo_csv[num_col])
#3.4
colomns_categorie = ["contrat_de_travail", "etat_civile", "Solvable"]
def encoder(colomns):
    le = preprocessing.LabelEncoder()
    le.fit(credits_immo_csv[colomns])
    list(le.classes_)
    #print(le.transform(credits_immo_csv[colomns]))
for i in colomns_categorie:
    #print(i)
    encoder(i)

#3.5
train, test = train_test_split(df, test_size=0.2)

#3.6
scaler = StandardScaler()
scaler.fit(credits_immo_csv[num_col])
scaler.mean_
scaler.transform(credits_immo_csv[num_col])


# 4.Visualisation de données

df = pd.read_csv("Data_Vis/Montant_temps.csv")
x = df.iloc[:,1]
y = df.iloc[:,0]
#4.1
plt.plot(x, y, color='red')
plt.xlabel('Temp') 
plt.title("Montant temps")   
plt.show()
#4.2
plt.scatter(x, y)
plt.show()

#5.Analyse de données

#5.1 
df = pd.read_csv("Data_Vis/tendance_centrale.csv")
df.mean()
df.median()
df.mode()
#5.2
df.boxplot()
model = ols('Age ~ Rating', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table
#5.3
df = pd.read_csv("Data_Vis/iris.csv")
df_corr = df.corr()
# faire tout les colomn df[["longueur_sepal", "largeur_sepal"]]
print(df_corr)
#5.3.1
x = df["longueur_sepal"]
y = df["longueur_petal"]



