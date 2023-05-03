from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from sklearn.model_selection import cross_val_score

def train_models(X_train, y_train, X_test, y_test):
    # Train and evaluate Multinomial Naive Bayes model
    mnb = MultinomialNB()
    mnb_scores = cross_val_score(mnb, X_train, y_train, cv=5)
    print("Multinomial Naive Bayes average score:", mnb_scores.mean())
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    print("Multinomial Naive Bayes evaluation:")
    print(classification_report(y_test, y_pred_mnb))

    # Train and evaluate k-Nearest Neighbors model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print("k-Nearest Neighbors average score:", knn_scores.mean())
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print("k-Nearest Neighbors evaluation:")
    print(classification_report(y_test, y_pred_knn))

    # Train and evaluate Logistic Regression model
    lr = LogisticRegression(penalty='l2', C=0.5, max_iter=1000)
    lr_scores = cross_val_score(lr, X_train, y_train, cv=5)
    print("Logistic Regression average score:", lr_scores.mean())
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    print("Logistic Regression evaluation:")
    print(classification_report(y_test, y_pred_lr))

    # Train and evaluate Support Vector Machine model(linear)
    svc = SVC(kernel='linear', C=0.08)
    svc_scores = cross_val_score(svc, X_train, y_train, cv=5)
    print("Support Vector Machine(linear) average score:", svc_scores.mean())
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    print("Support Vector Machine(linear) evaluation:")
    print(classification_report(y_test, y_pred_svc))

    # Train and evaluate Support Vector Machine model(RBF)
    svc = SVC(kernel='linear', C=10, gamma=0.10)
    svc_scores = cross_val_score(svc, X_train, y_train, cv=5)
    print("Support Vector Machine(RBF) average score:", svc_scores.mean())
    svc.fit(X_train, y_train)
    y_pred_svc = svc.predict(X_test)
    print("Support Vector Machine(RBF) evaluation:")
    print(classification_report(y_test, y_pred_svc))

    # Train and evaluate k-Means clustering model
    # kmeans = KMeans(n_clusters=2, random_state=42)
    # kmeans_scores = silhouette_score(X_train, kmeans.fit_predict(X_train))
    # print("k-Means clustering silhouette score:", kmeans_scores)
    # kmeans.fit(X_train, y_train)
    # y_pred_kmeans = kmeans.predict(X_test)
    # print("k-Means clustering evaluation:")
    # print(classification_report(y_test, y_pred_kmeans))
