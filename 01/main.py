import numpy as np

# Load the data
all_games = np.load('data.pkl',allow_pickle=True)
train_set = all_games['train_set']
test_set = all_games['test_set']


def erm_algorithm(training_set,prophets):
    min_rate_prophets = []
    best_prophet = 0
    prophet_num = 0
    erm = 1
    for prophet in prophets['train_set']:
        cur_erm = 0
        count =0
        for sample in training_set:
            if prophet[sample] != train_set[sample]:
                cur_erm +=1
            count +=1
        if cur_erm/count < erm:
            min_rate_prophets = [prophet_num]
            erm = cur_erm/count
            best_prophet = prophet_num
        elif cur_erm/count == erm:
            min_rate_prophets.append(prophet_num)
        prophet_num +=1
    best_prophet = np.random.choice(min_rate_prophets)
    return best_prophet,erm
def compute_avarage_error(prophet,test_set):
    count = 0
    error = 0
    for index in range(len(test_set)):
        if prophet[index] != test_set[index]:
            error +=1
        count +=1
    return error/count

def Scenario_1():
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    num_samples = 1
    prophets = np.load('scenario_one_and_two_prophets.pkl',allow_pickle=True)
    approximation_error = prophets['true_risk'].min()
    results = {'iteration':[],'erm':[],'average_error':[],'estimation_error':[],'approximation_error':[],'true_risk':[],'best_prophet':0}
    for i in range(100):
        training_set = np.random.choice(train_set.shape[0], num_samples)
        chosen_prophet,erm = erm_algorithm(training_set,prophets)
        average_error = compute_avarage_error(prophets['test_set'][chosen_prophet],test_set)
        estimation_error = prophets['true_risk'][chosen_prophet]-approximation_error
        if prophets['true_risk'][chosen_prophet] == approximation_error :
            results['best_prophet'] +=1
        results['iteration'].append(i+1)
        results['erm'].append(erm)
        results['average_error'].append(average_error)
        results['estimation_error'].append(estimation_error)
        results['approximation_error'].append(approximation_error)
        results['true_risk'].append(prophets['true_risk'][chosen_prophet])
    print(results)





def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    num_samples = 10
    prophets = np.load('scenario_one_and_two_prophets.pkl', allow_pickle=True)
    approximation_error = prophets['true_risk'].min()
    results = {'iteration': [], 'erm': [], 'average_error': [], 'estimation_error': [], 'approximation_error': [],
               'true_risk': [], 'best_prophet': 0}
    for i in range(100):
        training_set = np.random.choice(train_set.shape[0], num_samples)
        chosen_prophet, erm = erm_algorithm(training_set, prophets)
        average_error = compute_avarage_error(prophets['test_set'][chosen_prophet], test_set)
        estimation_error = prophets['true_risk'][chosen_prophet] - approximation_error
        if prophets['true_risk'][chosen_prophet] == approximation_error:
            results['best_prophet'] += 1
        results['iteration'].append(i + 1)
        results['erm'].append(erm)
        results['average_error'].append(average_error)
        results['estimation_error'].append(estimation_error)
        results['approximation_error'].append(approximation_error)
        results['true_risk'].append(prophets['true_risk'][chosen_prophet])
    print(results)


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    num_samples = 10
    prophets = np.load('scenario_three_and_four_prophets.pkl', allow_pickle=True)
    approximation_error = prophets['true_risk'].min()
    results = {'iteration': [], 'erm': [], 'average_error': [], 'estimation_error': [], 'approximation_error': [],
               'true_risk': [], 'best_prophet': 0,'less_than_1%':0}
    for i in range(100):
        training_set = np.random.choice(train_set.shape[0], num_samples)
        chosen_prophet, erm = erm_algorithm(training_set, prophets)
        average_error = compute_avarage_error(prophets['test_set'][chosen_prophet], test_set)
        estimation_error = prophets['true_risk'][chosen_prophet] - approximation_error
        if prophets['true_risk'][chosen_prophet] == approximation_error:
            results['best_prophet'] += 1
        if estimation_error < 0.01:
            results['less_than_1%'] +=1
        results['iteration'].append(i + 1)
        results['erm'].append(erm)
        results['average_error'].append(average_error)
        results['estimation_error'].append(estimation_error)
        results['approximation_error'].append(approximation_error)
        results['true_risk'].append(prophets['true_risk'][chosen_prophet])
    print(results)

def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    num_samples = 1000
    prophets = np.load('scenario_three_and_four_prophets.pkl', allow_pickle=True)
    approximation_error = prophets['true_risk'].min()
    results = {'iteration': [], 'erm': [], 'average_error': [], 'estimation_error': [], 'approximation_error': [],
               'true_risk': [], 'best_prophet': 0, 'less_than_1%': 0}
    for i in range(100):
        training_set = np.random.choice(train_set.shape[0], num_samples)
        chosen_prophet, erm = erm_algorithm(training_set, prophets)
        average_error = compute_avarage_error(prophets['test_set'][chosen_prophet], test_set)
        estimation_error = prophets['true_risk'][chosen_prophet] - approximation_error
        if prophets['true_risk'][chosen_prophet] == approximation_error:
            results['best_prophet'] += 1
        if estimation_error < 0.01:
            results['less_than_1%'] += 1
        results['iteration'].append(i + 1)
        results['erm'].append(erm)
        results['average_error'].append(average_error)
        results['estimation_error'].append(estimation_error)
        results['approximation_error'].append(approximation_error)
        results['true_risk'].append(prophets['true_risk'][chosen_prophet])
    print(results)


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    prophets = np.load('scenario_five_prophets.pkl', allow_pickle=True)
    num_of_prophets = [2, 5, 10,50]
    num_of_samples = [1,10, 50, 1000]
    for k in num_of_prophets:
        indices = np.random.choice(len(prophets['train_set']), k)
        current_train_sets = {'train_set': ([prophets['train_set'][i] for i in indices])}
        current_true_risks = {'true_risk': ([prophets['true_risk'][i] for i in indices])}
        approximation_error = round(np.min(current_true_risks['true_risk']),5)
        for samples_num in num_of_samples:
            av_err = 0
            av_est = 0
            for i in range(100):
                training_set = np.random.choice(train_set.shape[0], samples_num, replace=False)
                chosen_prophet, erm = erm_algorithm(training_set, current_train_sets)
                average_error = compute_avarage_error(prophets['test_set'][chosen_prophet], test_set)
                av_err += average_error
                estimation_error = prophets['true_risk'][chosen_prophet] - approximation_error
                av_est+=estimation_error
            av_err = abs(round(av_err/ 100,5))
            av_est = abs(round(av_est/100,5))
            print([k,samples_num])
            print([av_err,av_est,approximation_error])


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    num_samples = 10
    prophets = np.load('scenario_six_prophets.pkl', allow_pickle=True)
    hypo_1 = prophets['hypothesis1']
    hypo_2 = prophets['hypothesis2']
    approximation_error_1 = hypo_1['true_risk'].min()
    approximation_error_2= hypo_2['true_risk'].min()
    results_1 = {'iteration': [], 'erm': [], 'average_error': [], 'estimation_error': [], 'approximation_error': [],
               'true_risk': [], 'best_prophet': 0, 'less_than_1%': 0}
    results_2 = {'iteration': [], 'erm': [], 'average_error': [], 'estimation_error': [], 'approximation_error': [],
                 'true_risk': [], 'best_prophet': 0, 'less_than_1%': 0}
    av_err_1 = 0
    av_est_1 = 0
    av_err_2 = 0
    av_est_2 = 0
    for i in range(100):
        training_set= np.random.choice(train_set.shape[0], num_samples)
        chosen_prophet_1, erm = erm_algorithm(training_set, hypo_1)
        chosen_prophet_2, erm = erm_algorithm(training_set, hypo_2)
        average_error_1 = compute_avarage_error(hypo_1['test_set'][chosen_prophet_1], test_set)
        av_err_1 += average_error_1
        average_error_2 = compute_avarage_error(hypo_2['test_set'][chosen_prophet_2], test_set)
        av_err_2+= average_error_2
        estimation_error_1 = hypo_1['true_risk'][chosen_prophet_1] - approximation_error_1
        av_est_1 += estimation_error_1
        estimation_error_2 = hypo_2['true_risk'][chosen_prophet_2] - approximation_error_2
        av_est_2 += estimation_error_2
        if hypo_1['true_risk'][chosen_prophet_1] == approximation_error_1:
            results_1['best_prophet'] += 1
        if estimation_error_1 < 0.01:
            results_1['less_than_1%'] += 1
        results_1['iteration'].append(i + 1)
        results_1['erm'].append(erm)
        results_1['average_error'].append(average_error_1)
        results_1['estimation_error'].append(estimation_error_1)
        results_1['approximation_error'].append(approximation_error_1)
        results_1['true_risk'].append(hypo_1['true_risk'][chosen_prophet_1])
        if hypo_2['true_risk'][chosen_prophet_2] == approximation_error_2:
            results_2['best_prophet'] += 1
        if estimation_error_2 < 0.01:
            results_2['less_than_1%'] += 1
        results_2['iteration'].append(i + 1)
        results_2['erm'].append(erm)
        results_2['average_error'].append(average_error_2)
        results_2['estimation_error'].append(estimation_error_2)
        results_2['approximation_error'].append(approximation_error_2)
        results_2['true_risk'].append(hypo_2['true_risk'][chosen_prophet_2])
    av_err_1 = av_err_1/100
    av_est_1 = av_est_1/100
    av_err_2 = av_err_2/100
    av_est_2 = av_est_2/100
    print(["average error 1:", av_err_1, "estimation error 1:", av_est_1, "approximation error 1:", approximation_error_1])
    print(["average error 2:", av_err_2, "estimation error 2:", av_est_2, "approximation error 2:", approximation_error_2])




if __name__ == '__main__':
    

    # print(f'Scenario 1 Results:')
    # Scenario_1()
    #
    # print(f'Scenario 2 Results:')
    # Scenario_2()
    #
    # print(f'Scenario 3 Results:')
    # Scenario_3()
    #
    # print(f'Scenario 4 Results:')
    # Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()
    #
    # print(f'Scenario 6 Results:')
    # Scenario_6()

