import tqdm
import torch
import shifthappens.tasks.lost_in_translation.affine_transformations.affine as  a
import numpy as np
import math
import gc
import random

def eval_batched_numpy(data, model, eval_device, batch_size = 1000):
    if eval_device == data.device and data.shape[0] <= batch_size:
        total_num = data.shape[0]
        results = []
        with torch.no_grad():
            res = model(data).detach().cpu().numpy()
            return res
    else:
        total_num = data.shape[0]
        results = []
        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
        with torch.no_grad():
            for i, [data_slice] in enumerate(tqdm.tqdm(loader, leave=False, desc="eval model")):
                data_slice = data_slice.to(eval_device)
                res = model(data_slice).detach().cpu().numpy()
                # if (i // 10) == 0:
                #     gc.collect()
                results.append(res)
        return np.concatenate(results, axis=0)

def np_softmax(x):
    max = np.max(x,axis=1,keepdims=True) 
    e_x = np.exp(x - max)
    sum = np.sum(e_x,axis=1,keepdims=True)
    return e_x / sum 

def calculate_zoom_for_target(target, bounds, crops, start_coords):
    zooms_bounds, _ = a.get_zoom_bounds(crops, start_coords)
    zooms_current = a.get_zooms(bounds, crops)
    zooms_required = target / zooms_current
    zoom = torch.maximum(zooms_required, zooms_bounds).clamp(min=0.0, max=1.0)
    return zoom

def gather_masks_statistics(mask, bounds, crops):
    center = a.get_centers(bounds)
    center_m = a.get_centers(crops)
    center[:,0] = center[:,0] - center_m[:,0]
    center[:,1] = center[:,1] - center_m[:,1]

    zooms = a.get_zooms(bounds, crops)

    return {'center': center.numpy(), 'zoom': zooms.numpy()}

    # super slow currently
    # occupancy = a.calc_mask_occupancy(mask)

    # return {'center': center.numpy(), 'zoom': zooms.numpy(), 'occupancy': occupancy}

def rotation_linspace(model, model_name, data, eval_device, batch_size_model, batch_size_rotation, resolution, idx_fun=lambda x:x, do_resize=True, save_dir=None):

    if save_dir is not None:
        subdir = str(random.randint(1, 99999))
        from pathlib import Path
        p = Path(save_dir) / subdir
        p.mkdir(parents=False, exist_ok=True)
        exp_dir = str(p)
    else:
        exp_dir = None

    def do_loop(loop_i, model, model_name, eval_device, batch_size_model, resolution, idx_fun, do_resize, results_rotation, d):

        datapoint_zoom, cat, elem  = d
        imgs, masks, bounds, crops, start_coords = tuple(map(lambda x: x.to(eval_device), datapoint_zoom))

        target_zoom = 1./math.sqrt(2)

        zoom = calculate_zoom_for_target(target_zoom, bounds, crops, start_coords)

        max_zoom = 0.0
        zoomed = a.do_zoom(imgs, masks, bounds, crops, start_coords, zoom, max_zoom, verbose=False)
        imgs, masks, bounds, crops, start_coords = zoomed

        max_angle = 45./180.*np.pi
        angles = torch.minimum(a.find_max_angle(bounds, crops, start_coords, resize=do_resize), torch.tensor([max_angle], device=eval_device))
        steps = torch.linspace(0,resolution,resolution, dtype=torch.float32, device=eval_device)/resolution
        angles_steps = -angles + (2*angles * steps)

        num_rotation_l = math.ceil(angles_steps.shape[0] / batch_size_rotation)
        iterator = range(num_rotation_l)
        if num_rotation_l > 10:
            iterator = tqdm.tqdm(iterator, leave=False, desc='gen_data')
        res_rotation = []
        res_rotation_m = []
        res_rotation_b = []
        res_rotation_c = []
        for i in iterator:
            angles_slice = angles_steps[(batch_size_rotation*i):(batch_size_rotation*i+batch_size_rotation)]
            datas = a.expand_data(imgs, masks, bounds, crops, start_coords, angles_slice.shape[0])
            r_imgs, r_masks, r_bounds, r_crops, r_start_coords = a.rescale_cropped(*a.crop_batches(*a.rotate(*datas, angles_slice, max_angle, verbose=False)))
            res_rotation.append(r_imgs.cpu())
            res_rotation_m.append(r_masks.cpu())
            res_rotation_b.append(r_bounds.cpu())
            res_rotation_c.append(r_crops.cpu())
        del datas

        r_imgs = torch.cat(res_rotation, 0)
        r_masks = torch.cat(res_rotation_m, 0)
        r_bounds = torch.cat(res_rotation_b, 0)
        r_crops = torch.cat(res_rotation_c, 0)
            #rotated = a.rotate(*datas, angles_steps, max_angle, verbose=False)

            #cropped_imgs, cropped_masks = a.crop_batches(*rotated)
            #rescaled_imgs_rotated, rescaled_masks = a.rescale_cropped(cropped_imgs, cropped_masks)


        if eval_device.type == 'cuda':
            with torch.no_grad():
                pred_rotated = eval_batched_numpy(r_imgs, model, eval_device, batch_size=batch_size_model)
            softmaxed_rotated = np_softmax(pred_rotated)
        else:
                #a.plot_debug_random_pytorch(*res)
            pred_rotated = eval_batched_numpy(r_imgs, model, eval_device)
            softmaxed_rotated = np_softmax(pred_rotated)

        angles_cpu = angles_steps.cpu()
        
        if save_dir is not None:
            sample_p = save_sample_images(r_imgs, softmaxed_rotated, angles_cpu, cat, exp_dir, loop_i)
        else:
            sample_p = None

        res = {
                "model": model_name,
                "data": idx_fun(elem),
                "cat": cat,
                "params": angles_cpu.numpy(),
                "results" : softmaxed_rotated,
                "masks_stats": gather_masks_statistics(r_masks, r_bounds, r_crops),
                "sample_params": sample_p,
                "sample_idx": loop_i,
                "exp_dir":exp_dir
            }

        results_rotation.append(res)

    results_rotation = []

    for i,d in enumerate(tqdm.tqdm(data, desc=f"{model_name}:rotation")):
        do_loop(i,model, model_name, eval_device, batch_size_model, resolution, idx_fun, do_resize, results_rotation, d)
        gc.collect()

    return results_rotation

def save_sample_images(images, out, params, label: int, dir, idx, model_check=None):
    if type(images) is list:
        def get_p(i,j,idx):
            if isinstance(params, torch.Tensor):
                return params[i][j][idx]
            else:
                return tuple(p[i][j][idx] for p in params)

        miss_classes = []
        for i in range(len(out)):
            for j in range(len(out[i])):
                classification: np.ndarray = out[i][j].argmax(1)
                miss_class = np.where(classification != label)[0]
                if miss_class.shape[0] != 0:
                    w_sample_idx_idx = random.choice(range(miss_class.shape[0]))
                    w_sample_idx = miss_class[w_sample_idx_idx]
                    miss_classes.append((i,j,w_sample_idx))
        if len(miss_classes) == 0:
            w_sample_idx = None
            wrong_params = None
        else:
            w_sample_idx = random.choice(miss_classes)
            (i,j,w_arr_idx) = w_sample_idx
            wrong_sample = images[i][j][w_arr_idx].cpu().permute(1,2,0).numpy()
            wrong_params = get_p(i,j,w_arr_idx)
            assert label != out[i][j].argmax(1)[w_arr_idx]
            with open(f'{dir}/{idx}_wrong.npy', 'wb') as f_w:
                np.save(f_w, wrong_sample)
            if model_check is not None:
                test = np.load(f'{dir}/{idx}_wrong.npy')
                m_d = next(model_check.parameters()).device
                test_torch = torch.from_numpy(test).permute(-1,0,1).unsqueeze(0).to(m_d)
                res = model_check(test_torch)
                if label == res.argmax():
                    import debugpy

                    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
                    debugpy.listen(5678)
                    print("Waiting for debugger attach")
                    debugpy.wait_for_client()
                    debugpy.breakpoint()
                    print('break on this line')
                assert label != res.argmax()
        s_i = random.choice(range(len(images)))
        s_j = random.choice(range(len(images[s_i])))
        sample_arr_idx = random.choice(range(images[s_i][s_j].shape[0]))
        sample_idx = (s_i, s_j, sample_arr_idx)
        sample = images[s_i][s_j][sample_arr_idx].cpu().permute(1,2,0).numpy()
        sample_params = get_p(s_i, s_j, sample_arr_idx)
        with open(f'{dir}/{idx}.npy', 'wb') as f_s:
            np.save(f_s, sample)
    else:
        def get_p(idx):
            if isinstance(params, torch.Tensor):
                return params[idx].cpu().numpy()
            else:
                return tuple(p[idx].cpu().numpy() for p in params)
        classification = out.argmax(1)
        miss_class = np.where(classification != label)[0]
        if miss_class.shape[0] == 0:
            w_sample_idx = None
            wrong_params = None
        else:
            w_sample_idx_idx = random.choice(range(miss_class.shape[0]))
            w_sample_idx = miss_class[w_sample_idx_idx]
            wrong_sample = images[w_sample_idx].cpu().permute(1,2,0).numpy()
            wrong_params = get_p(w_sample_idx)
            assert label != classification[w_sample_idx]
            with open(f'{dir}/{idx}_wrong.npy', 'wb') as f_w:
                np.save(f_w, wrong_sample)
            if model_check is not None:
                test = np.load(f'{dir}/{idx}_wrong.npy')
                m_d = next(model_check.parameters()).device
                test_torch = torch.from_numpy(test).permute(-1,0,1).unsqueeze(0).to(m_d)
                res = model_check(test_torch)
                assert label != res.argmax()

        sample_idx = random.choice(range(images.shape[0]))
        sample = images[sample_idx].cpu().permute(1,2,0).numpy()
        sample_params = get_p(sample_idx)
        with open(f'{dir}/{idx}.npy', 'wb') as f_s:
            np.save(f_s, sample)
    
    return (sample_idx, sample_params), (w_sample_idx, wrong_params)