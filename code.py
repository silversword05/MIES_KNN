import glob
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

feature = []
target_names = ['sad', 'neutral', 'happy']
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
    return {'list': [], 'click': 0, 'right_click': 0, 'left_click': 0, 'time_last': 0.0, 'left_time_elapsed': 0.0, 'right_time_elapsed': 0.0,
            'scroll0_count': 0.0, 'scroll0_time': 0.0, 'scroll1_count': 0.0, 'scroll1_time': 0.0, 'scroll2_count': 0.0, 'scroll2_time': 0.0,
            'scroll0_time_list': [], 'scroll1_time_list': [], 'scroll2_time_list': [], 'right_click_time': [], 'left_click_time': [], 'x_cod': [], 'y_cod': [],
            'drag_x_cod': [], 'drag_y_cod': [], 'time': [], 'drag_time': [], 'speedX': [], 'speedY': [], 'accX': [], 'accY': []}


def feature_extractor(f_in, label, feature_function, labels_function, count):
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
        elif words[0] == "MD":
            state['drag_x_cod'].append(float(words[1]))
            state['drag_y_cod'].append(float(words[2]))
            state['drag_time'].append(float(words[3]))
        elif words[0] == 'MP' or words[0] == 'MR' or words[0] == 'MC' or words[0] == 'MWM':
            state['list'].append(words)
        elif words[0] == 'MC':
            if state['list'][len(state['list']) - 2][0] == 'MR':
                state['left_click'] += 1 if int(words[1]) == 1 else 0
                state['left_time_elapsed'] += int(state['list'][len(state['list']) - 2][2]) if int(words[1]) == 1 else 0
                state['left_click_time'].append(time_val - int(state['list'][len(state['list']) - 2][2])) if int(words[1]) == 1 else 0
                state['right_click'] += 1 if not int(words[1]) == 1 else 0
                state['right_time_elapsed'] += int(state['list'][len(state['list']) - 2][2]) if not int(words[1]) == 1 else 0
                state['right_click_time'].append(time_val - int(state['list'][len(state['list']) - 2][2])) if not int(words[1]) == 1 else 0
            state['click'] += 1
        elif words[0] == 'MWM':
            def assign_scroll_count(param_count, param_time):
                state[param_time] += 0 if state[param_count] == 0 else int(words[6])
                state[param_count] += 1

            if int(words[4]) == 0:
                assign_scroll_count('scroll0_count', 'scroll0_time')
            elif int(words[4]) == 1:
                assign_scroll_count('scroll1_count', 'scroll1_time')
            elif int(words[4]) == -1:
                assign_scroll_count('scroll2_count', 'scroll2_time')
        else:
            state['scroll0_time_list'].append(state['scroll0_time'])
            state['scroll1_time_list'].append(state['scroll1_time'])
            state['scroll2_time_list'].append(state['scroll2_time'])
            state['scroll0_count'] = state['scroll0_time'] = state['scroll1_count'] = state['scroll1_time'] = state['scroll2_count'] = state['scroll2_time'] = 0.0

        if count % 30 == 0 and count != 0 and count > 0:
            time_diff = time_val - time_last
            right_click_frequency = state['right_click'] / time_diff
            left_click_frequency = state['left_click'] / time_diff
            state['right_time_elapsed'] = 0 if state['right_click'] == 0 else state['right_time_elapsed'] / state['right_click']
            state['left_time_elapsed'] = 0 if state['left_click'] == 0 else state['left_time_elapsed'] / state['left_click']
            left_single_click, left_double_click = double_click_count(state['left_click_time'])
            right_single_click, right_double_click = double_click_count(state['right_click_time'])

            def calculate_differential(key, index_func):
                return (state[key][index_func + 1] - state[key][index_func]) / state['time'][index_func + 1]

            if len(state['time']) > 0:
                state['time'][0] = 0
                for val1, val2 in zip(['speedX', 'speedY', 'accX', 'accY'], ['x_cod', 'y_cod', 'speedX', 'speedY']):
                    state[val1].extend([calculate_differential(val2, index) for index in range(len(state[val2]) - 1) if state['time'][index + 1] != 0])

            speed_x_avg, speed_y_avg, acc_x_avg, acc_y_avg = average_scroll_time(state['speedX']), average_scroll_time(state['speedY']), average_scroll_time(
                state['accX']), average_scroll_time(state['accY'])

            drag_speed_x_avg, drag_speed_y_avg, drag_acc_x_avg, drag_acc_y_avg = average_scroll_time(state['speedX']), average_scroll_time(state['speedY']), average_scroll_time(
                state['accX']), average_scroll_time(state['accY'])

            feature_function.append(
                [time_diff, state['click'], time_diff, state['click'], right_click_frequency, state['right_time_elapsed'], right_single_click, right_double_click,
                 left_click_frequency, state['left_time_elapsed'], left_single_click, left_double_click, average_scroll_time(state['scroll0_time_list']),
                 average_scroll_time(state['scroll1_time_list']), average_scroll_time(state['scroll2_time_list']), speed_x_avg, speed_y_avg, acc_x_avg, acc_y_avg,
                 drag_speed_x_avg, drag_speed_y_avg, drag_acc_x_avg, drag_acc_y_avg])
            labels_function.append(label)

            state = get_new_state()
            count = 0
            time_last = time_val
        count += 1


def extract_features():
    global feature, cont_feature, labels, cont_labels
    users = ["data/17EC35025"]
    test_users = ["data/17EC35025"]
    for user, test_user in zip(users, test_users):
        sad_files = glob.glob(user + "/Emotional/Sad/*.txt")
        for f in sad_files:
            feature_extractor(f, 0, feature, labels, -6)
        neut_files = glob.glob(user + "/Neutral/*.txt")
        for f in neut_files:
            feature_extractor(f, 1, feature, labels, -6)
        happy_files = glob.glob(user + "/Emotional/Happy/*.txt")
        for f in happy_files:
            feature_extractor(f, 2, feature, labels, -6)
        cont_files = glob.glob(test_user + '/*.txt')
        for p, f in enumerate(sorted(cont_files)):
            feature_extractor(f, 1 - p, cont_feature, cont_labels, -9)  # ---------------   BE CAUTION OF THIS LINE p VALUE ---------------
        print(user)


def train_test_model():
    global feature, cont_feature, labels, cont_labels
    sc = StandardScaler()
    feature = sc.fit_transform(feature)
    cont_feature = sc.fit_transform(cont_feature)

    classifier = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(classifier, feature, labels, cv=5)
    print("Train Accuracy", end=' ')
    print(scores.mean())

    classifier.fit(feature, labels)
    y_pred = classifier.predict(cont_feature)
    cm = confusion_matrix(y_true=cont_labels, y_pred=y_pred)

    print("Test Accuracy", end=' ')
    print(accuracy_score(y_true=cont_labels, y_pred=y_pred))

    print("For the continuous data number of instances for each mood are predicted as:")
    print(cm)


extract_features()
train_test_model()
