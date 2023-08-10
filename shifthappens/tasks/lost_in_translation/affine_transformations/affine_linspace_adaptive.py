import tqdm
import torch
import shifthappens.tasks.lost_in_translation.affine_transformations.affine as  a
import shifthappens.tasks.lost_in_translation.affine_transformations.affine_linspace as  a_s
import numpy as np
import math
import gc
import random
import tqdm
import gc
import collections
from skimage.feature.peak import peak_local_max

def find_mins(res_correct, res_incorrect, past_size_x, past_size_y, nums, find_min_correct=True):
    if find_min_correct:
        mat_t: np.ndarray = res_correct.reshape(past_size_x, past_size_y)#.cpu().numpy()
        mat = (-1)*mat_t + np.max(mat_t)
        indices_unraveled = peak_local_max(mat, min_distance=3, exclude_border=False, num_peaks=nums)
    else:
        mat_correct: np.ndarray = res_correct.reshape(past_size_x, past_size_y)
        mat_res_incorrect: np.ndarray = res_incorrect.reshape(past_size_x, past_size_y)
        temp = mat_correct - mat_res_incorrect
        mat = (-1)*temp + np.max(temp)
        indices_unraveled = peak_local_max(mat, min_distance=3, exclude_border=False, num_peaks=nums)
    indices = np.ravel_multi_index((indices_unraveled[:,0], indices_unraveled[:,1]), mat.shape)
    return indices[:nums], (-1)*np.take(np.reshape(mat, -1), indices)

def sample_adaptive(trans_xs: list[torch.Tensor], trans_ys: list[torch.Tensor], softmaxes_s: list[np.ndarray],
    num_points, resolution, cat, bounds, size_recursive, eval_device, batch_size_translation, data, model,
    batch_size_model, past_t_x, past_t_y, find_min_correct=True, adapt_resolution=False):
    results = {}
    for idx in range(len(trans_xs)):
        softmaxes = softmaxes_s[idx]
        res_incorrect = np.max(np.delete(softmaxes, cat, axis=1), axis=1)
        res_correct = softmaxes[:, cat]
        past_size_x = past_t_x[idx].shape[0]
        past_size_y = past_t_y[idx].shape[0]
        idx_mins, min_vals = find_mins(res_correct, res_incorrect, past_size_x, past_size_y, num_points, find_min_correct=find_min_correct)
        for idx2 in range(idx_mins.shape[0]):
            results[min_vals[idx2]] = (idx, idx_mins[idx2])

    if len(results) == 0:
        for idx in range(len(trans_xs)):
            softmaxes = softmaxes_s[idx]
            res_correct = softmaxes[:, cat]

            idx_min = np.argmin(res_correct)
            results[res_correct[idx_min].item()] = (idx, idx_min)

    keys_sorted = list(sorted(results.keys()))

    move_max_below, move_max_above, move_max_left, move_max_right = bounds
    height = move_max_above + move_max_below
    if size_recursive < 1.0:
        area_height = size_recursive * height
    else:
        area_height = min(height.item(), size_recursive)
    width = move_max_right + move_max_left
    if size_recursive < 1.0:
        area_width = size_recursive * width
    else:
        area_width = min(width.item(), size_recursive)
    center = a.get_centers(data[2])
    results_imgs = []
    results_masks = []
    results_bounds = []
    results_crops = []
    results_pred_translated = []
    results_softmaxed_translated = []
    results_coords_x = []
    results_coords_y = []
    results_points = []
    results_t_x = []
    results_t_y = []

    for key in keys_sorted[:num_points]:
        (list_idx, min_idx) = results[key]
        x_coord = trans_xs[list_idx][min_idx]
        y_coord = trans_ys[list_idx][min_idx]
        results_points.append((x_coord.item(), y_coord.item(), list_idx))
        x_start = max(x_coord - (area_height / 2), (-1)*move_max_above)
        x_end = min(x_coord + (area_height / 2), move_max_below)
        y_start = max(y_coord - (area_width / 2), (-1)*move_max_left)
        y_end = min(y_coord + (area_width / 2), move_max_right)
        t_x, t_y = calculate_linspace(x_start, x_end, y_start, y_end, resolution, center, eval_device, adapt_resolution=adapt_resolution)

        combinations = torch.cartesian_prod(t_x, t_y)
        comb_x = combinations[:,0]
        comb_y = combinations[:,1]
        assert math.ceil(comb_x.shape[0] / min(batch_size_translation, batch_size_model)) == 1

        temp = translation_helper(model, comb_x, comb_y, batch_size_translation, batch_size_model, data, eval_device)
        r_imgs, r_masks, r_bounds, r_crops, pred_translated, softmaxed_translated = temp
        results_imgs.append(r_imgs)
        results_masks.append(r_masks)
        results_bounds.append(r_bounds)
        results_crops.append(r_crops)
        results_pred_translated.append(pred_translated)
        results_softmaxed_translated.append(softmaxed_translated)
        results_coords_x.append(comb_x)
        results_coords_y.append(comb_y)
        results_t_x.append(t_x)
        results_t_y.append(t_y)
    return results_imgs, results_masks, results_bounds, results_crops, results_pred_translated, results_softmaxed_translated, results_coords_x, results_coords_y, results_points, results_t_x, results_t_y 


def translation_helper(model, comb_x, comb_y, batch_size_translation, batch_size_model, data, eval_device, only_softmaxes=False):
    num_trans_l = math.ceil(comb_x.shape[0] / batch_size_translation)
    iterator = range(num_trans_l)
    res_translated = []
    res_translated_m = []
    res_translated_b = []
    res_translated_c = []

    overall_steps = math.ceil(comb_x.shape[0] / min(batch_size_translation, batch_size_model))

    if overall_steps > 1:
        iterator = tqdm.tqdm(iterator, leave=False, desc='gen_data')

        assert not only_softmaxes

        def do_inner_loop(i):
            comb_x_slice = comb_x[(batch_size_translation*i):(batch_size_translation*i+batch_size_translation)]
            comb_y_slice = comb_y[(batch_size_translation*i):(batch_size_translation*i+batch_size_translation)]
            datas = a.expand_data(*data, comb_x_slice.shape[0])
            r_imgs, r_masks, r_bounds, r_crops, r_start_coords = a.rescale_cropped(*a.crop_batches(*a.translate_xy(*datas, comb_x_slice, comb_y_slice, verbose=False)))
            res_translated.append(r_imgs.cpu())
            res_translated_m.append(r_masks.cpu())
            res_translated_b.append(r_bounds.cpu())
            res_translated_c.append(r_crops.cpu())


        for i in iterator:
            do_inner_loop(i)

        r_imgs = torch.cat(res_translated, 0)
        r_masks = torch.cat(res_translated_m, 0)
        r_bounds = torch.cat(res_translated_b, 0)
        r_crops = torch.cat(res_translated_c, 0)

        if eval_device.type == 'cuda':
            with torch.no_grad():
                pred_translated = a_s.eval_batched_numpy(r_imgs, model, eval_device, batch_size=batch_size_model)
                softmaxed_translated = a_s.np_softmax(pred_translated)
        else:
            pred_translated = a_s.eval_batched_numpy(r_imgs, model, eval_device)
            softmaxed_translated = a_s.np_softmax(pred_translated)
    
    else:
        with torch.no_grad():
            datas = a.expand_data(*data, comb_x.shape[0])
            def guard():
                r_imgs, r_masks, r_bounds, r_crops, r_start_coords = a.rescale_cropped(*a.crop_batches(*a.translate_xy(*datas, comb_x, comb_y, verbose=False)))
                if only_softmaxes:
                    return r_imgs, None, None, None
                else:
                    return r_imgs, r_masks.cpu(), r_bounds.cpu(), r_crops.cpu()
            r_imgs_cuda, r_masks, r_bounds, r_crops = guard()
            if eval_device.type == 'cuda':
                pred_translated = a_s.eval_batched_numpy(r_imgs_cuda, model, eval_device, batch_size=batch_size_model)
                softmaxed_translated = a_s.np_softmax(pred_translated)
            else:
                pred_translated = a_s.eval_batched_numpy(r_imgs_cuda, model, eval_device)
                softmaxed_translated = a_s.np_softmax(pred_translated)
            if only_softmaxes:
                r_imgs = None
            else:
                r_imgs = r_imgs_cuda.cpu()
                

    return r_imgs, r_masks, r_bounds, r_crops, pred_translated, softmaxed_translated

def calculate_linspace(start_x, end_x, start_y, end_y, resolution, center, eval_device, adapt_resolution=False):
    if resolution > 10:
        steps = torch.linspace(0,resolution,resolution, dtype=torch.float32, device=eval_device)/resolution
        
        t_x = start_x + (end_x - start_x) * steps
        t_y = start_y + (end_y - start_y) * steps
    else:
        center_x = center[:,0]
        center_y = center[:,1]

        def calculate_steps(start, end, center, step_size):
            offset = (center % step_size)
            #calculation
            #start2 = res*((start - offset) // res) + res*math.ceil(((start - offset) % res)) + offset
            real_start = start - offset
            start_i = torch.div(real_start, step_size, rounding_mode='trunc')
            start_i = start_i.int()
            real_end = end - offset
            end_i = torch.div(real_end, step_size, rounding_mode='floor')
            end_i = end_i.int()
            start_i = min(start_i, end_i)
            leng = end_i - start_i
            assert leng >= 0
            if (leng == 0).item():
                if adapt_resolution:
                    if step_size <= 0.05:
                        return torch.zeros_like(start_i).float() + start
                    else:
                        return calculate_steps(start, end, center, 1/2*step_size)
                else:
                    #next step is too much, maybe last step would be ok (we may skip the first)
                    last_step = (center % step_size) - step_size
                    if (last_step >= start and last_step <= end).item():
                        return last_step.float()
                    else:
                        return torch.zeros_like(start_i).float() + start
            else:
                steps_i =torch.arange(start=start_i.item(),end=end_i.item()+1e-4,step=1.0, dtype=torch.float, device=eval_device)
                steps = (step_size * steps_i.float()) + offset
                return steps
        t_x = calculate_steps(start_x, end_x, center_x, resolution)
        t_y = calculate_steps(start_y, end_y, center_y, resolution)

    return t_x,t_y

def calc_min_grid(start_x, end_x, start_y, end_y, step_size, lenght, eval_device, model, batch_size_translation, batch_size_model, data, cat, big_step_size):
    def calc_start_end(bound_lower,bound_upper):
        middle = bound_lower + (bound_upper - bound_lower)/2
        start = max(middle - float(lenght)/2,bound_lower)
        end = min(middle + float(lenght)/2,bound_upper)
        steps = torch.arange(start=start.item(), end=(end.item() + 1e-4), step=step_size, device=eval_device)
        return steps
    t_x = calc_start_end(start_x, end_x)
    t_x_l = t_x.shape[0]
    t_y = calc_start_end(start_y, end_y)
    t_y_l = t_y.shape[0]
    combinations = torch.cartesian_prod(t_x, t_y)
    comb_x = combinations[:,0]
    comb_y = combinations[:,1]
    out = translation_helper(model, comb_x, comb_y, batch_size_translation, batch_size_model, data, eval_device, only_softmaxes=True)
    r_imgs, r_masks, r_bounds, r_crops, pred_translated, softmaxed_translated = out
    res_correct = softmaxed_translated[:, cat]
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    params = (comb_x.cpu().numpy(), comb_y.cpu().numpy())
    X = np.stack(params, axis=1)
    regressor.fit(X, res_correct)
    pred_raw = regressor.predict(X)
    temp = (res_correct - pred_raw).reshape(t_x_l, t_y_l)
    indices_unraveled = peak_local_max((-1)*(temp + temp.max()), exclude_border=int(0.4/step_size), min_distance=int(0.4/step_size), num_peaks=4)
    if indices_unraveled.shape[0] == 0:
        idx_min = np.argmin(res_correct - pred_raw)
    else:
        idx_min = np.ravel_multi_index((indices_unraveled[:,0], indices_unraveled[:,1]), temp.shape)
        idx_min = torch.from_numpy(idx_min).to(eval_device)
    x_coords_min = comb_x[idx_min]
    y_coords_min = comb_y[idx_min]
    x_coord_min = (x_coords_min % big_step_size).mean()
    y_coord_min = (y_coords_min % big_step_size).mean()
    center = torch.stack((x_coord_min, y_coord_min)).view(1,2)
    min_grid_data = {
        "params": params,
        "results": softmaxed_translated,
        "t_x": t_x.cpu().numpy(),
        "t_y": t_y.cpu().numpy(),
        "point": center.cpu().numpy(),
        "points":(x_coords_min.cpu().numpy(), y_coords_min.cpu().numpy())
    }
    return center, min_grid_data


def translation_linspace_adaptive(model, model_name, data, eval_device, batch_size_model, batch_size_translation,
    resolutions, num_points, size_recursive, num_recursion, idx_fun=lambda x:x, target_zoom=0.8, save_dir = None,
    find_min_correct=True, early_stopping=False, adapt_grid_to_min=False, step_size_center=0.25, leng_center=4,
    period_assumption=1.0, constant_offset=None, adapt_resolution=False, check_saving=False):

    if save_dir is not None:
        subdir = str(random.randint(1, 99999))
        from pathlib import Path
        p = Path(save_dir) / subdir
        p.mkdir(parents=False, exist_ok=True)
        exp_dir = str(p)
    else:
        exp_dir = None

    def do_loop(i, model, model_name, d, results_translation):
        datapoint_zoom, cat, idx  = d
        imgs, masks, bounds, crops, start_coords = tuple(map(lambda x: x.to(eval_device), datapoint_zoom))

        zoom = a_s.calculate_zoom_for_target(target_zoom, bounds, crops, start_coords)

        max_zoom = 0.0
        zoomed = a.do_zoom(imgs, masks, bounds, crops, start_coords, zoom, max_zoom, verbose=False)
        imgs, masks, bounds, crops, start_coords = zoomed

        t_bounds = a.calculate_translate_bounds(crops, bounds, start_coords)
        move_max_below, move_max_above, move_max_left, move_max_right = t_bounds

        resolution = resolutions[0]

        if not adapt_grid_to_min:
            align_grid = a.get_centers(bounds)
            min_grid_data = None
        else:
            if constant_offset is None:
                align_grid, min_grid_data = calc_min_grid(-move_max_above, move_max_below, -move_max_left, move_max_right,
                    step_size_center, leng_center, eval_device, model, batch_size_translation, batch_size_model, zoomed, cat, period_assumption)
            else:
                align_grid = torch.from_numpy(constant_offset).view(1,2).float().to(imgs.device)
                min_grid_data = None
        t_x, t_y = calculate_linspace(-move_max_above, move_max_below, -move_max_left, move_max_right,
            resolution, align_grid, eval_device, adapt_resolution=adapt_resolution)

        if  len(t_x) == 0 and len(t_y) != 0:
            t_x = torch.tensor([0], device=eval_device, dtype=t_y.dtype)
        elif len(t_y) == 0 and len(t_x) != 0:
            t_y = torch.tensor([0], device=eval_device, dtype=t_x.dtype)
        

        if len(t_x) == 0 and len(t_y) == 0:
            fst = move_max_above.item() == 0 and move_max_below.item() == 0
            snd = move_max_below.item() == 0 and move_max_left.item() == 0
            assert (fst or snd) and resolution <= 10.0
            print("skipping")
            return {"error": "unable to discretize"}

        combinations = torch.cartesian_prod(t_x, t_y)
        comb_x = combinations[:,0]
        comb_y = combinations[:,1]

        out = translation_helper(model, comb_x, comb_y, batch_size_translation, batch_size_model, zoomed, eval_device)
        r_imgs, r_masks, r_bounds, r_crops, pred_translated, softmaxed_translated = out

        params_cpu = (comb_x.cpu(), comb_y.cpu())

        params_list_x=[[comb_x.cpu().numpy()]]
        params_list_y=[[comb_y.cpu().numpy()]]
        results_list=[[softmaxed_translated]]
        images_list = [[r_imgs]]
        masks_list = [[r_masks]]
        points_list = [[]]
        t_x_list = [[t_x.cpu().numpy()]]
        t_y_list = [[t_y.cpu().numpy()]]

        current_results = ([comb_x], [comb_y], [softmaxed_translated])
        if type(size_recursive) is list:
            current_size = size_recursive[0]
        else:
            current_size = 1.0

        zoomed_data = (imgs, masks, bounds, crops, start_coords)

        to_cpu_np = lambda l: list(map(lambda x: x.cpu().numpy(), l))

        def should_stop(results_softmaxs):
            if not early_stopping:
                return False
            else:
                for softm in results_softmaxs:
                    res_correct = softm[:, cat]
                    res_incorrect = np.max(np.delete(softm, cat, axis=1), axis=1)
                    if np.any(res_incorrect > res_correct):
                        return True
                return False

        past_t_x = [t_x.cpu().numpy()]
        past_t_y = [t_y.cpu().numpy()]
        if not should_stop([softmaxed_translated]):
            for num in range(num_recursion - 1):
                if type(size_recursive) is list:
                    current_size = size_recursive[num]
                else:
                    current_size = current_size * size_recursive
                r_i_l, r_m_l, r_b_l, r_c_l, r_p_l, r_s_l, r_x, r_y, r_p, t_x_l, t_y_l = sample_adaptive(*current_results,
                    num_points, resolutions[num+1], cat, t_bounds, current_size, eval_device, batch_size_translation, zoomed_data, model,
                    batch_size_model, past_t_x, past_t_y, find_min_correct=find_min_correct, adapt_resolution=adapt_resolution)
                current_results = (r_x, r_y, r_s_l)
                params_list_x.append(to_cpu_np(r_x))
                params_list_y.append(to_cpu_np(r_y))
                results_list.append(r_s_l)
                images_list.append(r_i_l)
                masks_list.append(r_m_l)
                points_list.append(r_p)
                t_x_list.append(to_cpu_np(t_x_l))
                t_y_list.append(to_cpu_np(t_y_l))
                past_t_x = t_x_l
                past_t_y = t_y_l
                if should_stop(r_s_l):
                    break

        params_tuple = (params_list_x, params_list_y)

        if save_dir is not None:
            model_to_check = None
            if check_saving:
                model_to_check = model
            sample_p = a_s.save_sample_images(images_list, results_list, params_tuple, cat, exp_dir, i, model_check=model_to_check)
        else:
            sample_p = None

        res = {
            "model": model_name,
            "data": idx_fun(idx),
            "cat": cat,
            "params": params_tuple,
            "points": points_list,
            "results" : results_list,
            "masks_stats": a_s.gather_masks_statistics(r_masks, r_bounds, r_crops),
            "sample_params": sample_p,
            "sample_idx": i,
            "t_x": t_x_list,
            "t_y": t_y_list,
            "min_grid_data": min_grid_data,
            "exp_dir": exp_dir,
            "batch_size_translation": batch_size_translation
        }

        results_translation.append(res)

    results_translation = []

    for i, d in enumerate(tqdm.tqdm(data, desc=f"{model_name}:translation")):
        do_loop(i, model, model_name, d, results_translation)
        gc.collect()

    return results_translation