from sklearn import svm

def build_svm():
    model = svm.SVC(kernel='linear')
    return model
