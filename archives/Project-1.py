#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[13]:


# load data
train_data = pd.read_csv('/Users/evatoledano/ML 2/project/train.csv')

train_data.head()


# In[56]:


train_data.shape


# In[14]:


train_data.info()


# 
# ## infos pour nous
# 
# - Id : Un identifiant unique pour chaque observation. Ce n'est probablement pas utile pour la prédiction et peut être ignoré dans l'analyse.
# Variables Physiques et Géographiques :
# - Elevation : L'élévation du terrain (probablement en mètres ou en pieds).
# - Aspect : Orientation azimutale du versant (en degrés).
# - Slope : Pente du terrain (en degrés).
# Horizontal_Distance_To_Hydrology : Distance horizontale à la source d'eau la plus proche.
# - Vertical_Distance_To_Hydrology : Distance verticale à la source d'eau la plus proche.
# - Horizontal_Distance_To_Roadways : Distance horizontale aux routes les plus proches.
# - Hillshade_9am/Noon/3pm : Mesure de l'ombre à différents moments de la journée.
# - Horizontal_Distance_To_Fire_Points : Distance horizontale aux points de feu (zones d'incendie) les plus proches.
# Ces variables sont susceptibles d'être fortement corrélées avec le type de couverture forestière en raison de leur impact direct sur l'écologie locale.
# - Variables de Zone Sauvage (Wilderness_Area1 à Wilderness_Area4) : Variables catégorielles (probablement codées en one-hot) indiquant dans quelle zone de nature sauvage se trouve l'observation. Elles peuvent être importantes car différentes zones peuvent avoir des caractéristiques écologiques distinctes.
# - Variables de Type de Sol (Soil_Type1 à Soil_Type40) : Similaire aux zones sauvages, ces variables indiquent le type de sol et sont également catégorielles, probablement codées en one-hot. Le sol influence grandement la végétation et donc le type de couverture forestière.
# - Cover_Type : Variable cible pour la classification, indiquant le type de couverture forestière.
# 

# In[39]:


train_data['Cover_Type'].describe()


# In[40]:


train_data.describe()


# In[16]:


plt.figure(figsize=(10, 6))
train_data['Elevation'].hist(bins=50)
plt.title('Distribution of Elevation')
plt.xlabel('Elevation')
plt.ylabel('Frequency')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
train_data['Aspect'].hist(bins=50)
plt.title('Distribution of Aspect')
plt.xlabel('Aspect')
plt.ylabel('Frequency')
plt.show()


# In[20]:


missing_values = train_data.isnull().sum()

print(missing_values)

# no missing values


# In a classification problem, such as predicting forest cover type, it is crucial to know whether the training data are balanced between the different classes: for the : Model Performance and Balancing Strategies

# In[24]:


# check the balance class
# Cover_Type -> variable that we want to predict

train_data.Cover_Type.value_counts()

#-> everything is balanced


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Cover_Type', data=train_data)

mean_value = train_data['Cover_Type'].mean()
median_value = train_data['Cover_Type'].median()

plt.axvline(x=mean_value, color='g', linestyle='--', label='mean')
plt.axvline(x=median_value, color='r', linestyle='-', label='median')

plt.legend()
plt.title('Distribution of the Target: Cover_Type')
plt.xlabel('Cover_Type')
plt.ylabel('Count')

plt.show()


# In[75]:


# Convert 'Cover_Type' to a numeric type if it represents numbers
train_data['Cover_Type'] = train_data['Cover_Type'].astype(int)

# Then you can plot the KDE
plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_data, x='Cover_Type', hue='Wilderness_Area', fill=True)

plt.title('Density of Cover Types Among Different Wilderness Areas')
plt.xlabel('Cover Type')
plt.ylabel('Density')

plt.show()


# ## Infos pour nous
# 
# - Courbes colorées : Chaque couleur représente une zone de nature sauvage différente (1 à 4). L'aire sous chaque courbe indique la probabilité estimée de la densité de la couverture forestière pour chaque zone de nature sauvage.
# - Pics des courbes : Les endroits où les courbes atteignent un pic indiquent la concentration la plus élevée de données pour un type de couverture forestière donné dans une zone de nature sauvage spécifique. Par exemple, un pic élevé pour la zone de nature sauvage 1 sur le type de couverture 2 indiquerait que ce type de couverture est particulièrement commun dans cette zone.
# 
# 
# Le graphique permet d'observer les différences dans les distributions des types de couverture entre les différentes zones de nature sauvage. Par exemple, si une couleur a un pic très prononcé par rapport aux autres, cela indiquerait que le type de couverture correspondant est plus prévalent dans cette zone de nature sauvage particulière par rapport aux autres

# In[76]:


selected_columns = ['Cover_Type', 'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                    'Horizontal_Distance_To_Fire_Points']
selected_data = train_data[selected_columns]

correlations = selected_data.corr()

plt.figure(figsize=(12, 10))  # Ajuster la taille selon vos besoins
sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# Une couleur rouge indique une corrélation positive, une couleur bleue indique une corrélation négative, et l'intensité de la couleur indique la force de la corrélation. Par exemple, une couleur rouge foncé indique une corrélation positive forte, tandis qu'une couleur bleue foncée indique une corrélation négative forte.
# Valeurs de Corrélation: Les nombres à l'intérieur de chaque cellule montrent la valeur exacte du coefficient de corrélation, allant de -1 à 1. Un 1 signifie une corrélation positive parfaite, un -1 signifie une corrélation négative parfaite, et un 0 indique l'absence de corrélation.
# 
# 
# Interprétation:
# 
# Des valeurs proches de 1 ou -1 indiquent que les variables sont fortement corrélées positivement ou négativement. Par exemple, Horizontal_Distance_To_Roadways et Elevation ont une corrélation de 0.58, ce qui suggère une corrélation positive modérément forte.
# Des valeurs proches de 0 indiquent une faible ou aucune corrélation linéaire. Par exemple, Aspect et Cover_Type ont une corrélation de 0.01, indiquant une très faible corrélation linéaire.

# In[32]:


correlations_target = train_data.corrwith(train_data['Cover_Type']).sort_values(ascending=False)
print(correlations_target)


# In[33]:


#Correlations with the Cover_Type variable


correlations_target = train_data.corrwith(train_data['Cover_Type']).sort_values(ascending=False)


plt.figure(figsize=(10, 8)) 
sns.barplot(x=correlations_target.values, y=correlations_target.index)


plt.title('Correlations with the Cover_Type variable')
plt.xlabel('Correlation coefficient')
plt.ylabel('Variables')
plt.show()


# In[67]:


#Visualization of the distribution of numerical variables

train_data.hist(bins=15, figsize=(30, 20), layout=(8, 7))

plt.figure(figsize=(30, 20))


# In[38]:


# correlations between Physical and geographical variables and covertypes

variables = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
             'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
             'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
             'Horizontal_Distance_To_Fire_Points']


plt.figure(figsize=(20, 20))

for i, var in enumerate(variables):
    plt.subplot(5, 2, i + 1)
    for cover_type in train_data['Cover_Type'].unique():
        subset = train_data[train_data['Cover_Type'] == cover_type]
        sns.kdeplot(subset[var], label=f'Cover Type {cover_type}', shade=True)
    plt.title(f'Distribution of {var} by Cover Type')
    plt.xlabel(var)
    plt.ylabel('Density')
    plt.legend()

plt.tight_layout()
plt.show()


# In[50]:


import pandas as pd
import matplotlib.pyplot as plt



wilderness_columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']


wilderness_counts = train_data[wilderness_columns].sum()


wilderness_ratios = wilderness_counts / len(train_data)


wilderness_df = pd.DataFrame({'Count': wilderness_counts, 'Ratio': wilderness_ratios})


ax = wilderness_df['Count'].plot(kind='bar', color='pink', figsize=(10, 6), alpha=0.7)


for i, ratio in enumerate(wilderness_df['Ratio']):
    ax.text(i, wilderness_df['Count'][i], f'{ratio:.2f}', ha='center', va='bottom')


plt.title('Number of Observations in Wilderness Areas')
plt.xlabel('Wilderness Area')
plt.ylabel('Count')
plt.xticks(rotation=0)  


plt.show()
##


# ## pour mieux comprendre
# 
# Graphique des Wilderness Areas
# 
# Le premier graphique montre le nombre d'observations (Count) pour chacune des quatre zones de nature sauvage dans le jeu de données. Les hauteurs des barres indiquent le nombre d'observations dans chaque zone. Les chiffres au-dessus des barres sont les ratios, qui représentent la proportion des observations dans chaque zone de nature sauvage par rapport au nombre total d'observations. Par exemple, si l'une des barres a un ratio de 0.24, cela signifie que 24% de toutes les observations du jeu de données se trouvent dans cette zone de nature sauvage particulière.
# 
# L'intérêt de ce graphique est de montrer comment les observations sont réparties parmi les différentes zones de nature sauvage, ce qui peut être pertinent car les caractéristiques écologiques peuvent varier considérablement d'une zone à l'autre et influencer le type de couverture forestière.

# In[51]:


soil_type_columns = [f'Soil_Type{i}' for i in range(1, 41)]


soil_type_counts = train_data[soil_type_columns].sum()


soil_type_ratios = soil_type_counts / len(train_data)


soil_type_df = pd.DataFrame({'Count': soil_type_counts, 'Ratio': soil_type_ratios}).sort_values(by='Count', ascending=False)


ax = soil_type_df['Count'].plot(kind='bar', color='lightgreen', figsize=(15, 8), alpha=0.7)


for i, ratio in enumerate(soil_type_df['Ratio']):
    ax.text(i, soil_type_df['Count'][i], f'{ratio:.2f}', ha='center', va='bottom')


plt.title('Number of Observations for Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.xticks(rotation=90)  


plt.tight_layout()  
plt.show()


# ## Pour mieux comprendre
# 
# Graphique des Soil Types
# 
# Le deuxième graphique présente le nombre d'observations pour chaque type de sol. Comme pour le premier graphique, les hauteurs des barres indiquent le nombre d'observations et les chiffres annotés sont les ratios. Ces ratios montrent la fréquence relative de chaque type de sol dans le jeu de données.
# 
# Dans ce graphique, vous pouvez voir que certains types de sol sont beaucoup plus fréquents que d'autres. Par exemple, un type de sol avec un ratio de 0.14 est présent dans 14% des observations, ce qui le rend potentiellement très significatif pour la prédiction du type de couverture forestière. D'autres types de sol avec des ratios très faibles peuvent être moins importants ou plus rares dans la région étudiée.

# Interprétation
# 
# L'interprétation de ces graphiques peut vous aider à comprendre l'importance relative de différentes caractéristiques écologiques et environnementales dans votre jeu de données. En connaissant la répartition des zones de nature sauvage et des types de sol, vous pouvez déduire des informations sur :
# 
# La diversité des zones de nature sauvage et des types de sol dans le jeu de données.
# Les types de sol ou les zones de nature sauvage qui pourraient avoir une influence plus forte sur le type de couverture forestière en raison de leur prévalence.
# Les types de sol ou les zones de nature sauvage qui pourraient nécessiter une attention particulière dans l'analyse en raison de leur rareté.
# Ces informations peuvent être utilisées pour affiner vos modèles de machine learning, par exemple, en donnant plus de poids aux caractéristiques qui sont moins fréquentes mais potentiellement très informatives pour prédire le type de couverture forestière.

# ## Test data

# In[53]:


# load data
test_data = pd.read_csv('/Users/evatoledano/ML 2/project/test-full.csv')

test_data.head()


# In[57]:


test_data.shape


# In[54]:


test_data.describe()


# # difference between test data et train data

# In[78]:


print(train_data.shape)
print(test_data.shape)


# In[80]:


# Compare the number of observations in test versus train data
train_size = train_data.shape[0]  
test_size = test_data.shape[0]    

# Data to plot
sizes = [train_size, test_size]
labels = ['Training Data', 'Test Data']

sns.barplot(x=labels, y=sizes)
plt.title('Train vs Test Data Sizes')
plt.ylabel('Number of Observations')
plt.show()


# **Data Exploration**

# Our dataset is composed of 1 target variable and 54 features, both categorical and numerical:
# 
# Numerical features:
# - **Elevation**, quantitative (meters): Elevation in meters
# - **Aspect**, quantitative (azimuth): Aspect in degrees azimuth
# - **Slope**, quantitative (degrees): Slope in degrees
# - **Horizontal_Distance_To_Hydrology**, quantitative (meters): Horz Dist to nearest surface water features
# - **Vertical_Distance_To_Hydrology**, quantitative (meters): Vert Dist to nearest surface water features
# - **Horizontal_Distance_To_Roadways**, quantitative (meters ): Horz Dist to nearest roadway
# - **Hillshade_9am**, quantitative (0 to 255 index): Hillshade index at 9am, summer solstice
# - **Hillshade_Noon**, quantitative (0 to 255 index): Hillshade index at noon, summer soltice
# - **Hillshade_3pm**, quantitative (0 to 255 index): Hillshade index at 3pm, summer solstice
# - **Horizontal_Distance_To_Fire_Points**, quantitative (meters): Horz Dist to nearest wildfire ignition points
# 
# Categorical features:
# - **Wilderness_Area** (4 binary columns), qualitative (0 (absence) or 1 (presence)): Wilderness area designation
# - **Soil_Type** (40 binary columns), qualitative (0 (absence) or 1 (presence)): Soil Type designation
# 
# Target:
# - **Cover_Type** (7 types), integer (1 to 7): Forest Cover Type designation <br><br><br>
# 
# 
# 
# The target variable (cover_type) is categorical and includes 7 possibilities:
# 1. Spruce/Fir
# 2. Lodgepole Pine
# 3. Ponderosa Pine
# 4. Cottonwood/Willow
# 5. Aspen
# 6. Douglas-fir
# 7. Krummholz <br><br><br>
# 
# There are no missing values, and all the values types are integers so no additional pre-processing is required at this stage.

# # Feature engeniring 

# In[4]:


from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_numerical_columns(df):
    
    df_scaled = df.copy()
    
    # List of columns not to be normalized
    non_scaled_columns = ['Id', 'Cover_Type'] + [f'Soil_Type{i}' for i in range(1, 41)] + [f'Wilderness_Area{i}' for i in range(1, 5)]
    
    # List of columns to be normalized
    scaled_columns = [col for col in df_scaled.columns if col not in non_scaled_columns]
    
 
    scaler = StandardScaler()
    
    df_scaled[scaled_columns] = scaler.fit_transform(df_scaled[scaled_columns])
    
    return df_scaled




# In[ ]:





# # test model

# In[1]:


# test model


from ast import literal_eval
from catboost import CatBoostClassifier
#from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier



def evaluate_models(X, y):
    models = {
        "LogReg": LogisticRegression(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "XGB": XGBClassifier(),
        "Catboost": CatBoostClassifier(verbose=0),
        #"LightGBM": LGBMClassifier(),
    }
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy') 
        results[name] = scores
        print(f"{name}: Accuracy = {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return results


# In[2]:





# In[ ]:




