from kliep import KLIEP
import numpy as np
"""
function performance：
1.importance reweighting (IW) method implementation
reference：Liu T, Tao D. Classification with Noisy Labels by Importance Reweighting[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2016, 38(3):447-461.
"""



def cal_cond_proba(train, sigma_chosen=1):
    """The density ratio is obtained by KLIEP"""

    def cal_density_ratio(train, label=-1, sigma_chosen=1):
        model = KLIEP()
        sub_train = train[train[:, 0] == label, 1:]
        density_ratio, _ = model.fit(sub_train.T, train[:, 1:].T, sigma_chosen=sigma_chosen)

        return density_ratio

    density_ratio_ne = cal_density_ratio(train, -1, sigma_chosen=sigma_chosen) # y=-1
    density_ratio_po = cal_density_ratio(train, 1, sigma_chosen=sigma_chosen)  # y=1

    cond_proba = []
    density_y = cal_density_Y(train)
    posi_index = 0
    nega_index = 0
    for i in range(train.shape[0]):
        if train[i, 0] == 1:
            cond_proba.append(density_ratio_po[posi_index] * density_y[0])
            posi_index += 1
        elif train[i, 0] == -1:
            cond_proba.append(density_ratio_ne[nega_index] * density_y[1])
            nega_index += 1
    cond_proba = np.array(cond_proba)
    cond_proba = np.reciprocal(cond_proba)
    return cond_proba



def cal_density_Y(train):
    """
    The result of Y=+1 is in the first place, and the result of Y=-1 is in the second place
    :param train: training set,Y is part of {+1,-1}
    :return: The density of y
    """

    density_y = []
    sample_nums = train.shape[0]

    posi_num = np.count_nonzero(train[:, 0] == 1)
    nega_num = np.count_nonzero(train[:, 0] == -1)

    density_y.append(float(posi_num / sample_nums))
    density_y.append(float(nega_num / sample_nums))
    return density_y




def cal_sample_weight(data, labels_noise, sigma_chosen=1):
    """
    Calculate the sample_weight in importance reweighting
    :param data: ndarray, training set,Y is part of {+1,-1}
    :param labels_noise: tuple, (the positive noise, the negative noise)
    :param sigma_chosen: int, kernel width 
    :return: sample_weight
    """""

    condition_proba = cal_cond_proba(data, sigma_chosen=sigma_chosen)   # Calculate conditional probability
    nu = []    # 计算分子
    de = []    # 计算分母

    temp = 1 - labels_noise[0] - labels_noise[1]
    for i in range(len(condition_proba)):

        if condition_proba[i] != 0:
            if data[i, 0] == 1:
                nu.append(condition_proba[i] - labels_noise[1])
                de.append(temp * condition_proba[i])

            elif data[i, 0] == -1:
                nu.append(condition_proba[i] - labels_noise[0])
                de.append(temp * condition_proba[i])
        else:   # 若条件概率为0，则beta则为0
            nu.append(0)
            de.append(1)

    nu = np.array(nu)
    de = np.array(de)
    sample_weight = nu / de
    return sample_weight