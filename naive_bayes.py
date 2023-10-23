from math import log10
import bayes_data_import

S = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
L = ['e', 'j', 's']

def get_training_data(files):
    train_x, train_y, freq = [], [], []
    for filename in files:
        x, y, f = bayes_data_import.read_file_from_name(filename)
        train_x.append(x)
        train_y.append(y)
        freq.append(f)
    
    return train_x, train_y, freq

def prior(training_data, y, smooth):
    K_L = len(L)
    _, train_y, _ = training_data
    count = 0
    for label in train_y:
        if label == y: count += 1
    
    p = (count + smooth) / (len(train_y) + K_L * smooth)

    return p

def likelihood(training_data, x, y, smooth):
    K_S = len(S)
    count = 0
    full_count = 0
    _, train_y, train_freq = training_data
    for i in range(len(train_y)):
        label, f = train_y[i], train_freq[i]
        if label != y: continue
        count += f[-1 if c == ' ' else ord(c) - ord('a')]
        full_count += sum(f)
    
    p_c_y = (count + smooth) / (full_count + K_S * smooth)
    return p_c_y

def evaluate(files, theta_e, theta_j, theta_s):
    output = []
    for filename in files:
        _, label, f = get_training_data([filename])
        f = f[0]
        label = label[0]
        log_p_x_e, log_p_x_j, log_p_x_s = 0.0, 0.0, 0.0
        for c in S:
            index = -1 if c == ' ' else ord(c) - ord('a')
            log_p_x_e += (f[index]) * log10(theta_e[index])
            log_p_x_j += (f[index]) * log10(theta_j[index])
            log_p_x_s += (f[index]) * log10(theta_s[index])
        
        probs = [(log_p_x_e, 'e'), (log_p_x_j, 'j'), (log_p_x_s, 's')]
        pred = max(probs, key = lambda x: x[0])[-1]
        output.append([pred, label])
    
    return output


if __name__ == '__main__':
    training_files = []

    for i in range(10):
        training_files.append('e' + str(i) + '.txt')
        training_files.append('j' + str(i) + '.txt')
        training_files.append('s' + str(i) + '.txt')
    
    training_data = get_training_data(training_files)

    print('1. Prior Probabilities:')
    print('-----------------------')

    print('English Prior Prob. -', round(prior(training_data, 'e', 0.5), 5))
    print('Japanese Prior Prob. -', round(prior(training_data, 'j', 0.5), 5))
    print('Spanish Prior Prob. -', round(prior(training_data, 's', 0.5), 5))
    print()

    print('2. Conditional Probabilites: English')
    print('------------------------------------')

    theta_e = []
    for c in S:
        theta_e.append(round(likelihood(training_data, c, 'e', 0.5), 5))
    
    print('theta_e -', theta_e)
    print()

    print('3. Conditional Probabilities: Japanese')
    print('--------------------------------------')

    theta_j = []
    for c in S:
        theta_j.append(round(likelihood(training_data, c, 'j', 0.5), 5))
    
    print('theta_j -', theta_j)
    print()

    print('3. Conditional Probabilities: Spanish')
    print('-------------------------------------')

    theta_s = []
    for c in S:
        theta_s.append(round(likelihood(training_data, c, 's', 0.5), 5))
    
    print('theta_s -', theta_s)
    print()

    print('4. Bag of Words Count Vector - e10.txt:')
    print('---------------------------------------')

    _, _, f = get_training_data(['e10.txt'])
    f = f[0]
    print('Bag-of-words vector -', f)
    print()

    print('5. Conditional Probabilities for e10.txt:')
    print('-----------------------------------------')

    log_p_x_e, log_p_x_j, log_p_x_s = 0.0, 0.0, 0.0
    for c in S:
        index = -1 if c == ' ' else ord(c) - ord('a')
        log_p_x_e += (f[index]) * log10(theta_e[index])
        log_p_x_j += (f[index]) * log10(theta_j[index])
        log_p_x_s += (f[index]) * log10(theta_s[index])
    
    print('The Conditional Probabilities with log_10 are -', round(log_p_x_e, 5), round(log_p_x_j, 5), round(log_p_x_s, 5))
    print()

    print('6. Prediction:')
    print('--------------')

    # As \hat{p}(x) is same, we can use log_p_x_y as proxy for a-posteriori probabilities.
    # Without \hat{p}(x) we can only calculate \hat{p}(x, y) concretely and not \hat{p}(y | x)

    probs = [(log_p_x_e, 'e'), (log_p_x_j, 'j'), (log_p_x_s, 's')]
    print('Posterior probabilities are -', round(log_p_x_e + log10(1/3), 5), round(log_p_x_j + log10(1/3), 5), round(log_p_x_s + log10(1/3), 5))
    print('The predicted class is -', max(probs, key = lambda x: x[0])[-1])
    print()

    print('7. Confusion Matrix:')
    print('--------------------')

    testing_files = []
    for i in range(10, 20):
        testing_files.append('e' + str(i) + '.txt')
        testing_files.append('j' + str(i) + '.txt')
        testing_files.append('s' + str(i) + '.txt')

    d = {'e': 0, 'j': 1, 's': 2}

    result = evaluate(testing_files, theta_e, theta_j, theta_s)
    confusion_matrix = [[0]*3 for _ in range(3)]
    for pred, label in result:
        confusion_matrix[d[pred]][d[label]] += 1
    
    print(*confusion_matrix[0], sep = ' ')
    print(*confusion_matrix[1], sep = ' ')
    print(*confusion_matrix[2], sep = ' ')