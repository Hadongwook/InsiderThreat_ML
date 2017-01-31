import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans


user = pd.read_csv('C:/Users/lab/InsiderThreat/daily_work/daily_f_u354.csv', sep=',')
user = pd.DataFrame(user)
user_t = user.fillna(method='ffill', axis=1)

cols = user_t.columns.tolist()
cols = cols[2:-1] + cols[-1:]
user_t = user_t[cols]
#Kmeans 로 클러스터링 3개로 분류
kmeans_model = KMeans(n_clusters=5)
kmeans = kmeans_model.fit(user_t)
labels = kmeans_model.labels_

# Create a PCA model.
pca_2 = PCA(2)
# Fit the PCA model on the numeric columns from earlier.
plot_columns = pca_2.fit_transform(user_t)
# Make a scatter plot of each game, shaded according to cluster assignment.
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels, edgecolors='white', s=100)
# Show the plot.
print(labels)
plt.show()
