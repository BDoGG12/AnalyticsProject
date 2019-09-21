# Starbucks location
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

game = pd.read_csv('nba.games.stats.csv', sep=',', index_col=0)

game.info()
game.head()

game.isnull().sum()

sns.heatmap(game.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.heatmap(game.corr(), cmap='coolwarm')

sns.set_style('whitegrid')
sns.jointplot(x='FieldGoals.', y='X3PointShots.', data=game)

game['OpponentPoints'].describe()

topD = game[game['OpponentPoints']<= 95]
worstD = game[game['OpponentPoints']>= 112]

sns.jointplot(x='Opp.Steals', y='Turnovers', data=game)

sns.lmplot(x='Opp.Steals', y='Turnovers', data=worstD)

game['Team'].unique()
game['Team'].nunique()

# Look at individual teams

okc = game[game['Team']=='OKC']
gs = game[game['Team']=='GSW']
bulls = game[game['Team']=='CHI']

sns.lmplot(x='FieldGoals.', y='X3PointShots.', data=okc)

okc['X3PointShots.'].describe()
okc['TeamPoints'].describe()
okc['OpponentPoints'].describe()

sns.scatterplot(x='Opp.FieldGoals', y='OpponentPoints', data=okc, hue='WINorLOSS')

sns.scatterplot(x='FieldGoals.', y='TeamPoints', data=okc, hue='WINorLOSS')

sns.jointplot(x='X3PointShots', y='X3PointShotsAttempted', data=okc)
sns.lmplot(x='X3PointShotsAttempted', y='X3PointShots', data=okc, 
           palette='coolwarm', size=6, aspect=1, fit_reg=False)

sns.lmplot(x='X3PointShots', y='X3PointShotsAttempted', data=okc,
           palette='coolwarm', size=6, aspect=1, fit_reg=False)

sns.lmplot(x='TeamPoints', y='OpponentPoints', hue='WINorLOSS',
           data=okc, size=6, aspect=1, palette='coolwarm',
           fit_reg=False)

sns.set_style('darkgrid')
g = sns.FacetGrid(okc, hue='WINorLOSS', size=6, aspect=2, palette='coolwarm')
g = g.map(plt.hist, 'TeamPoints', bins=20, alpha=0.7)

sns.lmplot(x='Blocks',y='Opp.FieldGoals.', palette='coolwarm',
           size=6, aspect=1, data=okc)

sns.lmplot(x='OffRebounds', y='TeamPoints', palette='coolwarm', hue='WINorLOSS',
           size=6, aspect=1, data=okc)

sns.lmplot(x='Assists', y='Turnovers', palette='coolwarm', hue='WINorLOSS',
           data=okc, size=6, aspect=1)

okc['Year'] = pd.to_datetime(okc['Date'], 
       errors='coerce').dt.to_period('Y')

sns.countplot(x='WINorLOSS', data=okc, hue='Home')

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()

okc['WINorLOSS'] = number.fit_transform(okc['WINorLOSS'].astype('str'))
okc['Home'] = number.fit_transform(okc['Home'].astype('str'))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


okc.columns
X = okc[[ 'Home', 'FieldGoals.', 'X3PointShots.', 'FreeThrows.', 'OffRebounds', 
         'TotalRebounds', 'Assists', 'Steals', 'Blocks', 
         'Turnovers', 'TotalFouls']]

y = okc['WINorLOSS']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,
                                                    random_state=101)

logmodel = LogisticRegression()

# Standardizing the data
from sklearn.preprocessing import StandardScaler
StandardScaler()
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))

cnf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions))

print(cnf_matrix)

sns.heatmap(cnf_matrix, annot=True, cmap='summer', cbar=False)
plt.title('Win / Loss Subscribed Confusion Matrix Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# accuracy score
accuracy_score(y_test, predictions)
logmodel.score(X,y)

# Coefficients of our logistic regression model
coeff_df = pd.DataFrame(list(zip(X.columns, np.transpose(logmodel.coef_))))
coeff_df.columns = ['Variables','Coefficient']
