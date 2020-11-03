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


# sad
def feature_extractor(f, label, feature, labels, count):
    file = open(f, 'r')
    List = []
    Time = 0
    click = 0
    right_click = 0
    left_click = 0
    # count=-12
    time_last = 0.0
    total_time_elapsed = 0.0
    left_time_elapsed = 0.0
    right_time_elapsed = 0.0
    scroll0_count = 0.0
    scroll0_time = 0.0
    scroll1_count = 0.0
    scroll1_time = 0.0
    scroll2_count = 0.0
    scroll2_time = 0.0
    scroll0_time_list = []
    scroll1_time_list = []
    scroll2_time_list = []
    right_click_time = []
    left_click_time = []
    x_cod = []
    y_cod = []
    drag_x_cod = []
    drag_y_cod = []
    time = []
    drag_time = []
    speedX = []
    speedY = []
    accX = []
    accY = []

    for line in file.readlines():
        words = str(line).split(',')
        if count != 0 and count > 0:
            try:
                Time += int(words[len(words) - 1])
            except:
                pass
        # count+=1
        if words[0] == "MM":
            x_cod.append(float(words[1]))
            y_cod.append(float(words[2]))
            time.append(float(words[3]))
        if words[0] == "MD":
            drag_x_cod.append(float(words[1]))
            drag_y_cod.append(float(words[2]))
            drag_time.append(float(words[3]))
        if words[0] == 'MP' or words[0] == 'MR' or words[0] == 'MC' or words[0] == 'MWM':
            List.append(words)
        if words[0] == 'MC':
            if int(words[1]) == 1:
                if List[len(List) - 2][0] == 'MR':
                    left_click += 1
                    left_time_elapsed += int(List[len(List) - 2][2])
                    left_click_time.append(Time - int(List[len(List) - 2][2]))
            else:
                if List[len(List) - 2][0] == 'MR':
                    right_click += 1
                    right_time_elapsed += int(List[len(List) - 2][2])
                    right_click_time.append(Time - int(List[len(List) - 2][2]))
            click += 1
        if words[0] == 'MWM':
            if int(words[4]) == 0:
                if scroll0_count == 0:
                    scroll0_time += 0
                    scroll0_count += 1
                else:
                    scroll0_time += int(words[6])
                    scroll0_count += 1
            if int(words[4]) == 1:
                if scroll1_count == 0:
                    scroll1_time += 0
                    scroll1_count += 1
                else:
                    scroll1_time += int(words[6])
                    scroll1_count += 1
            if int(words[4]) == -1:
                if scroll2_count == 0:
                    scroll2_time += 0
                    scroll2_count += 1
                else:
                    scroll2_time += int(words[6])
                    scroll2_count += 1
        else:
            scroll0_time_list.append(scroll0_time)
            scroll1_time_list.append(scroll1_time)
            scroll2_time_list.append(scroll2_time)
            scroll0_count = 0.0
            scroll0_time = 0.0
            scroll1_count = 0.0
            scroll1_time = 0.0
            scroll2_count = 0.0
            scroll2_time = 0.0
        if count % 30 == 0 and count != 0 and count > 0:
            TIME = Time - time_last
            right_click_frequency = right_click / TIME
            left_click_frequency = left_click / TIME
            try:
                right_time_elapsed = right_time_elapsed / right_click
            except:
                right_time_elapsed = 0
            try:
                left_time_elapsed = left_time_elapsed / left_click
            except:
                left_time_elapsed = 0
            Click = click
            left_single_click, left_double_click = double_click_count(left_click_time)
            right_single_click, right_double_click = double_click_count(right_click_time)
            # print len(time)
            # time[0]=0
            if len(time) > 0:
                time[0] = 0
                for index in range(len(x_cod) - 1):
                    if (time[index + 1] != 0):
                        speedX.append((x_cod[index + 1] - x_cod[index]) / time[index + 1])

                for index in range(len(y_cod) - 1):
                    if (time[index + 1] != 0):
                        speedY.append((x_cod[index + 1] - x_cod[index]) / time[index + 1])

                for index in range(len(speedX) - 1):
                    if (time[index + 1] != 0):
                        accX.append((speedX[index + 1] - speedX[index]) / time[index + 1])

                for index in range(len(speedY) - 1):
                    if (time[index + 1] != 0):
                        accY.append((speedY[index + 1] - speedY[index]) / time[index + 1])

            SpeedX_avg = average_scroll_time(speedX)
            SpeedY_avg = average_scroll_time(speedY)
            AccX_avg = average_scroll_time(accX)
            AccY_avg = average_scroll_time(accY)

            drag_SpeedX_avg = average_scroll_time(speedX)
            drag_SpeedY_avg = average_scroll_time(speedY)
            drag_AccX_avg = average_scroll_time(accX)
            drag_AccY_avg = average_scroll_time(accY)
            node = []
            node.append(TIME)
            node.append(click)
            node.append(right_click_frequency)
            node.append(right_time_elapsed)
            node.append(right_single_click)
            node.append(right_double_click)
            node.append(left_click_frequency)
            node.append(left_time_elapsed)
            node.append(left_single_click)
            node.append(left_double_click)
            node.append(average_scroll_time(scroll0_time_list))
            node.append(average_scroll_time(scroll1_time_list))
            node.append(average_scroll_time(scroll2_time_list))
            node.append(SpeedX_avg)
            node.append(SpeedY_avg)
            node.append(AccX_avg)
            node.append(AccY_avg)
            node.append(drag_SpeedX_avg)
            node.append(drag_SpeedY_avg)
            node.append(drag_AccX_avg)
            node.append(drag_AccY_avg)
            feature.append(node)
            labels.append(label)
            right_click_time = []
            left_click_time = []
            List = []
            click = 0
            right_click = 0
            left_click = 0
            count = 0
            time_last = Time
            total_time_elapsed = 0.0
            left_time_elapsed = 0.0
            right_time_elapsed = 0.0
            scroll0_time_list = []
            scroll1_time_list = []
            scroll2_time_list = []
            scroll0_count = 0.0
            scroll0_time = 0.0
            scroll1_count = 0.0
            scroll1_time = 0.0
            scroll2_count = 0.0
            scroll2_time = 0.0
            x_cod = []
            y_cod = []
            drag_x_cod = []
            drag_y_cod = []
            time = []
            drag_time = []
            speedX = []
            speedY = []
            accX = []
            accY = []
        count += 1


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

# sad_files=glob.glob(PATH + '/Emotional/Sad/*.txt')
# contusers = ["data/14EC32010/MouseLogDir", "data/15EC10020/MouseLogDir", "data/15EC10031/MouseLogDir","data/15EC10037/MouseLogDir", "data/15EC10051/MouseLogDir","data/Praneeth/MouseLogDir","data/Mohith/MouseLogDir",]
# users = ["data/14EC32010", "data/15EC10020", "data/15EC10031","data/15EC10037", "data/15EC10051", "data/Praneeth", "data/Mohith"]
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
