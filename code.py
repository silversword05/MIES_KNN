import glob
from typing import List
import numpy as np

feature = []
labels = []
cont_feature = []
cont_labels = []
file_count = 0


def double_click_count(time_list: List):
    if len(time_list) == 0:
        return 0, 0
    diff = [i - j for i, j in zip(time_list[:-1], time_list[1:])]
    count = sum([i <= 200 for i in diff]) * 2
    return len(time_list) - count, count


def average_scroll_time(scroll_time_list):
    if len(scroll_time_list) == 0:
        return 0
    return np.average(scroll_time_list)


def get_new_state():
    state_dict = {'list': [], 'click': 0, 'right_click': 0, 'left_click': 0, 'time_last': 0.0, 'left_time_elapsed': 0.0, 'right_time_elapsed': 0.0,
                  'scroll0_count': 0.0, 'scroll0_time': 0.0, 'scroll1_count': 0.0, 'scroll1_time': 0.0, 'scroll2_count': 0.0, 'scroll2_time': 0.0,
                  'scroll0_time_list': [], 'scroll1_time_list': [], 'scroll2_time_list': [], 'right_click_time': [], 'left_click_time': [], 'x_cod': [], 'y_cod': [],
                  'drag_x_cod': [], 'drag_y_cod': [], 'time': [], 'drag_time': [], 'speedX': [], 'speedY': [], 'accX': [], 'accY': []}
    return state_dict


# sad
def feature_extractor(f_in, label, feature, labels, count):
    file = open(f_in, 'r')
    time_val = 0
    time_last = 0.0

    state = get_new_state()

    for line in file.readlines():
        words = str(line).split(',')
        if count != 0 and count > 0:
            try:
                time_val += int(words[len(words) - 1])
            except:
                pass
        if words[0] == "MM":
            state['x_cod'].append(float(words[1]))
            state['y_cod'].append(float(words[2]))
            state['time'].append(float(words[3]))
        if words[0] == "MD":
            state['drag_x_cod'].append(float(words[1]))
            state['drag_y_cod'].append(float(words[2]))
            state['drag_time'].append(float(words[3]))
        if words[0] == 'MP' or words[0] == 'MR' or words[0] == 'MC' or words[0] == 'MWM':
            state['list'].append(words)
        if words[0] == 'MC':
            if int(words[1]) == 1:
                if state['list'][len(state['list']) - 2][0] == 'MR':
                    state['left_click'] += 1
                    state['left_time_elapsed'] += int(state['list'][len(state['list']) - 2][2])
                    state['left_click_time'].append(time_val - int(state['list'][len(state['list']) - 2][2]))
            else:
                if state['list'][len(state['list']) - 2][0] == 'MR':
                    state['right_click'] += 1
                    state['right_time_elapsed'] += int(state['list'][len(state['list']) - 2][2])
                    state['right_click_time'].append(time_val - int(state['list'][len(state['list']) - 2][2]))
            state['click'] += 1
        if words[0] == 'MWM':
            if int(words[4]) == 0:
                if state['scroll0_count'] == 0:
                    state['scroll0_time'] += 0
                    state['scroll0_count'] += 1
                else:
                    state['scroll0_time'] += int(words[6])
                    state['scroll0_count'] += 1
            if int(words[4]) == 1:
                if state['scroll1_count'] == 0:
                    state['scroll1_time'] += 0
                    state['scroll1_count'] += 1
                else:
                    state['scroll1_time'] += int(words[6])
                    state['scroll1_count'] += 1
            if int(words[4]) == -1:
                if state['scroll2_count'] == 0:
                    state['scroll2_time'] += 0
                    state['scroll2_count'] += 1
                else:
                    state['scroll2_time'] += int(words[6])
                    state['scroll2_count'] += 1
        else:
            state['scroll0_time_list'].append(state['scroll0_time'])
            state['scroll1_time_list'].append(state['scroll1_time'])
            state['scroll2_time_list'].append(state['scroll2_time'])
            state['scroll0_count'] = 0.0
            state['scroll0_time'] = 0.0
            state['scroll1_count'] = 0.0
            state['scroll1_time'] = 0.0
            state['scroll2_count'] = 0.0
            state['scroll2_time'] = 0.0
        if count % 30 == 0 and count != 0 and count > 0:
            time_diff = time_val - time_last
            right_click_frequency = state['right_click'] / time_diff
            left_click_frequency = state['left_click'] / time_diff
            try:
                state['right_time_elapsed'] = state['right_time_elapsed'] / state['right_click']
            except:
                state['right_time_elapsed'] = 0
            try:
                state['left_time_elapsed'] = state['left_time_elapsed'] / state['left_click']
            except:
                state['left_time_elapsed'] = 0
            left_single_click, left_double_click = double_click_count(state['left_click_time'])
            right_single_click, right_double_click = double_click_count(state['right_click_time'])
            if len(state['time']) > 0:
                state['time'][0] = 0
                for index in range(len(state['x_cod']) - 1):
                    if state['time'][index + 1] != 0:
                        state['speedX'].append((state['x_cod'][index + 1] - state['x_cod'][index]) / state['time'][index + 1])

                for index in range(len(state['y_cod']) - 1):
                    if state['time'][index + 1] != 0:
                        state['speedY'].append((state['x_cod'][index + 1] - state['x_cod'][index]) / state['time'][index + 1])

                for index in range(len(state['speedX']) - 1):
                    if state['time'][index + 1] != 0:
                        state['accX'].append((state['speedX'][index + 1] - state['speedX'][index]) / state['time'][index + 1])

                for index in range(len(state['speedY']) - 1):
                    if state['time'][index + 1] != 0:
                        state['accY'].append((state['speedY'][index + 1] - state['speedY'][index]) / state['time'][index + 1])

            speed_x_avg = average_scroll_time(state['speedX'])
            speed_y_avg = average_scroll_time(state['speedY'])
            acc_x_avg = average_scroll_time(state['accX'])
            acc_y_avg = average_scroll_time(state['accY'])

            drag_speed_x_avg = average_scroll_time(state['speedX'])
            drag_speed_y_avg = average_scroll_time(state['speedY'])
            drag_acc_x_avg = average_scroll_time(state['accX'])
            drag_acc_y_avg = average_scroll_time(state['accY'])
            feature.append([time_diff, state['click'], time_diff, state['click'], right_click_frequency, state['right_time_elapsed'], right_single_click, right_double_click,
                            left_click_frequency, state['left_time_elapsed'], left_single_click, left_double_click, average_scroll_time(state['scroll0_time_list']),
                            average_scroll_time(state['scroll1_time_list']), average_scroll_time(state['scroll2_time_list']), speed_x_avg, speed_y_avg, acc_x_avg, acc_y_avg,
                            drag_speed_x_avg, drag_speed_y_avg, drag_acc_x_avg, drag_acc_y_avg])
            labels.append(label)

            state = get_new_state()
            count = 0
            time_last = time_val
        count += 1


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

users = ["data/17EC35025"]
contusers = ["data/17EC35025"]
for user, contuser in zip(users, contusers):
    sad_files = glob.glob(user + "/Emotional/Sad/*.txt")
    for f in sad_files:
        feature_extractor(f, 0, feature, labels, -6)

    # neut_files=glob.glob(PATH+'/Neutral/*.txt')
    neut_files = glob.glob(user + "/Neutral/*.txt")
    for f in neut_files:
        feature_extractor(f, 1, feature, labels, -6)

    # happy_files=glob.glob(PATH+'/Emotional/Happy/*.txt')
    happy_files = glob.glob(user + "/Emotional/Happy/*.txt")
    for f in happy_files:
        feature_extractor(f, 2, feature, labels, -6)

    # cont_feature=[]
    # cont_labels =[]
    cont_files = glob.glob(contuser + '/*.txt')
    for p, f in enumerate(sorted(cont_files)):
        feature_extractor(f, 1 - p, cont_feature, cont_labels, -9)

    print(user)

# print(feature)
# print(cont_feature)
print(labels)
print(cont_labels)
# exit(0)

sc = StandardScaler()
feature = sc.fit_transform(feature)
cont_feature = sc.fit_transform(cont_feature)

# classifier = SVC(kernel='linear', random_state=0)
classifier = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(classifier, feature, labels, cv=5)
print("Accuracy in the Emotional and neutral data:")
print(scores.mean())

classifier.fit(feature, labels)
y_pred = classifier.predict(cont_feature)

cm = confusion_matrix(y_true=cont_labels, y_pred=y_pred)
print(accuracy_score(y_true=cont_labels, y_pred=y_pred))

target_names = ['sad', 'neutral', 'happy']

# print cm

# print "Accuracy score:"
# print accuracy_score(cont_labels, y_pred)


# exit(0)

print("For the continuous data number of instances for each mood are predicted as:")
print(cm)

# feature=[]
# labels=[]
# cont_feature=[]
# cont_labels =[]


##############PCA####################
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(feature)
X = pca.transform(feature)
print(X.shape)

x_cord = X[:, 0]
y_cord = X[:, 1]
z_cord = X[:, 2]

colors = ['red', 'green', 'blue']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_cord, y_cord, z_cord, c=labels)  # , label=[0, 1, 2])
plt.show()
