import numpy as np
import math

label_map = None

def max_freedom(d):
    t_x = d["t_x"][0][0]
    t_x_l = t_x[-1]-t_x[0]
    t_y = d["t_y"][0][0]
    t_y_l = t_y[-1]-t_y[0]
    return max(t_x_l,t_y_l)

def get_correct_class(d):
    i = d['data']
    cat_id = label_map[i]
    return cat_id

def has_wrong_classf(d):
    idx = get_correct_class(d)
    resu = d['results']
    has_wrong = False
    if 'min_grid_data' in d and d['min_grid_data'] is not None:
        min_data_res = d['min_grid_data']['results']
        res_correct = min_data_res[:, idx]
        res_incorrect = np.max(np.delete(min_data_res, idx, axis=1), axis=1)
        has_wrong = has_wrong or np.any(res_incorrect > res_correct)

    if type(resu) is list:
        for rec in resu:
            for point in rec:
                res_correct = point[:, idx]
                res_incorrect = np.max(np.delete(point, idx, axis=1), axis=1)
                has_wrong = has_wrong or np.any(res_incorrect > res_correct)
    else:
        res_correct = resu[:, idx]
        res_incorrect = np.max(np.delete(resu, idx, axis=1), axis=1)
        has_wrong = has_wrong or np.any(res_incorrect > res_correct)
    return has_wrong

def adaptive_worst_case(result):
    wrong_counter = 0
    for d in result:
        has_wrong = has_wrong_classf(d)
        if has_wrong:
            wrong_counter += 1
    return (len(result) - wrong_counter)/len(result)

def is_correct(d, mode):
    cat_id = get_correct_class(d)
    resu = d['results']
    if type(resu) is list:
        assert mode == "trans"
        t_x = d['params'][0][0][0]
        t_y = d['params'][1][0][0]
        min_idx = np.argmin((t_x**2)+(t_y**2))
        cat_pred = np.argmax(d['results'][0][0][min_idx])
        return cat_id == cat_pred
    else:
        if mode == "trans":
            t_x,t_y = d['params']
            min_idx = np.argmin((t_x**2)+(t_y**2))
            cat_pred = np.argmax(d['results'][min_idx])
        elif mode == "rotation":
            rot = d['params']
            min_idx = np.argmin(np.abs(rot))
            cat_pred = np.argmax(d['results'][min_idx])
        elif mode == "zoom":
            zoom = d['params']
            min_idx = np.max(zoom)
            cat_pred = np.argmax(d['results'][min_idx])
        return cat_id == cat_pred

def adaptive_base_case(result, mode):
    wrong_counter = 0
    for d in result:
        has_wrong = not is_correct(d, mode)
        if has_wrong:
            wrong_counter += 1
    return (len(result) - wrong_counter)/len(result)

def radiant_to_degree(data):
    return data * (180./math.pi)

def gt_30(d):
    return np.max(radiant_to_degree(d['params'])) >= 30