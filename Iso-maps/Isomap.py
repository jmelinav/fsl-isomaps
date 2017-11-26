from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

def isomap_dim_reduction(data,dimension,e=10):
    df = pd.DataFrame(data)
    distance_mat = generate_distance_matrix(df,e)
    path_matrix = floydWarshall(distance_mat)
    reduced_data = mds(path_matrix,dimension)
    return reduced_data


def plot_graph(data,colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=colors)
    ax.set_axis_off()
    plt.show()

def plot_graph2d(data,colors):
    sp = pl.subplot(211)
    U = np.dot(data, [[-.79, -.59, -.13],
                      [.29, -.57, .75],
                      [-.53, .56, .63]])
    sp.scatter(U[:, 1], U[:, 2], c=colors)
    sp.set_title("Original data")
    pl.show()

def plot_graph2d_n(data,colors):
    sp = pl.subplot(211)
    U = np.dot(data, [[-.79, -.59, -.13],
                      [.29, -.57, .75],
                      [-.53, .56, .63]])
    sp.scatter(U[:, 0], U[:, 2],  c=colors)
    sp.set_title("Original data")
    pl.show()

def plot_graph_3d_with_axis(data,colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.zaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')
    ax.w_xaxis.set_pane_color((0.1, 0.2, 0.5, 0.5))
    ax.w_yaxis.set_pane_color((0.1, 0.2, 0.5, 0.5))
    ax.w_zaxis.set_pane_color((0.1, 0.2, 0.5, 0.5))
    ax.scatter(data[:,0], data[:,1], data[:,2], c=colors)
    ax.set_axis_off()
    plt.show()


def reduce_swiss_roll():
    n = 500
    data,t = swiss_roll(n)
    t = t.astype(int)
    #data = twin_peaks()
    # df = pd.DataFrame(data)
    # df.columns = ['x','y','z']
    # df['color'] = t*2-1
    # df = df.loc[df['color']<9]
    # data=df.as_matrix()
    kmeans = KMeans(n_clusters=5,init="k-means++", n_init = 10, max_iter = 300, tol = 0.0001, precompute_distances ="auto", verbose = 0, random_state = None, copy_x = True, n_jobs = 1, algorithm ="auto")
    base = 5
    manifold = np.vstack((base*np.round((t * 2 - 1)/base), data[:,2])).T.copy()
    kmeans.fit(data)
    colors = kmeans.labels_#np.floor(manifold[:, 0])
    plot_graph(data[:,[0,1,2]],colors)
    reduced = isomap_dim_reduction(data,3)
    plot_graph(reduced,colors)
    reduced_df = pd.DataFrame(reduced)
    reduced_df[3] = colors
    train,test = separete_data_for_classification(80,reduced_df.as_matrix())
    knn_classifier(train,test,1)
    calculate_trustworthiness(reduced[:,[0,1]],data,7,n)
    calculate_continuity(reduced[:,[0,1]],data,7,n)

def reduce_helix():
    n = 500
    data,t = helix(n)
    t = t.astype(int)
    #data = twin_peaks()
    # df = pd.DataFrame(data)
    # df.columns = ['x','y','z']
    # df['color'] = t*2-1
    # df = df.loc[df['color']<9]
    # data=df.as_matrix()
    kmeans = KMeans(n_clusters=5,init="k-means++", n_init = 10, max_iter = 300, tol = 0.0001, precompute_distances ="auto", verbose = 0, random_state = None, copy_x = True, n_jobs = 1, algorithm ="auto")
    base = 5
    manifold = np.vstack((base*np.round((t * 2 - 1)/base), data[:,2])).T.copy()
    kmeans.fit(data)
    colors = kmeans.labels_#np.floor(manifold[:, 0])
    plot_graph(data[:,[0,1,2]],colors)
    reduced = isomap_dim_reduction(data,3,5)
    plot_graph(reduced,colors)
    reduced_df = pd.DataFrame(reduced)
    reduced_df[3] = colors
    train,test = separete_data_for_classification(80,reduced_df.as_matrix())
    knn_classifier(train,test,1)
    calculate_trustworthiness(reduced,data,7,n)
    calculate_continuity(reduced,data,7,n)

def knn_classifier(train,test,n):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(train[:,[0,1]],train[:,3])
    labels = knn.predict(test[:,[0,1]])
    correct = sum(labels == test[:, 3])
    #print('correct:',correct)
    number_of_test_samples = len(test)
    print('Classification generalisation error: ',
          (number_of_test_samples - correct)*100 /number_of_test_samples,'%')

#reduce_swiss_roll()
reduce_helix()
# train_samples = random.sample(range(100), 20)
# print(train_samples)
# print(np.around(train_samples,decimals=-1))


