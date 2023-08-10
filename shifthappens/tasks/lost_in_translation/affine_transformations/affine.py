from pickle import TRUE
import torch
import numpy as np
from skimage.transform import resize
import skimage.io as io
import kornia.geometry.transform as t
import torch.nn as nn
import random
import matplotlib.pyplot as plt
import torch.functional as F

config_imagenet = {
    'target_size': 254,
    'crop_size': 254,
    #'add_space_factor': 1 #left,right,upper,lower
}

config = config_imagenet


def square_dim(old_start, old_length, new_length, space, oversized_allowed=False):
    space_left = old_start
    space_right = space - (old_start + old_length)
    half_space_needed_float = (new_length - old_length)/2.0
    half_space_needed = int(half_space_needed_float)
    assert half_space_needed >= 0

    if space_left > half_space_needed_float and space_right > half_space_needed_float:
        new_start = old_start - half_space_needed
    elif space_left < half_space_needed_float and space_right > (half_space_needed_float + half_space_needed - space_left):
        new_start = 0
    elif space_left > (half_space_needed_float + half_space_needed_float - space_right) and space_right < half_space_needed_float:
        new_start = old_start - ((new_length - old_length) - space_right)
    else:
        if not oversized_allowed:
            raise Exception("unable to compute square")
        else:
            new_start = old_start - half_space_needed
    return new_start, new_length

def square_bbox(coco_bbox, h, w, oversized_allowed=False):
    bbox_x, bbox_y, bbox_w, bbox_h = map(int,coco_bbox)
    if bbox_w > bbox_h:
        bbox_y, bbox_h = square_dim(bbox_y, bbox_h, bbox_w, h, oversized_allowed=oversized_allowed)
    elif bbox_w < bbox_h:
        bbox_x, bbox_w = square_dim(bbox_x, bbox_w, bbox_h, w, oversized_allowed=oversized_allowed)
    
    return int(bbox_x), int(bbox_w), int(bbox_y), int(bbox_h)

def get_bounds(img, do_check=True):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    if do_check:
        assert (rmax - rmin) == (cmax - cmin)

    return rmin, rmax, cmin, cmax

def load_and_pad_imagenet(I: np.ndarray, gt_uint: np.ndarray, in_label, load_other_masks_fun=None, accept_smaller=True):
    m = np.where(gt_uint == in_label)

    bbox_y = np.min(m[0])
    bbox_h = np.max(m[0]) - np.min(m[0])
    bbox_x = np.min(m[1])
    bbox_w = np.max(m[1]) - np.min(m[1])
    bbox = (bbox_x, bbox_y, bbox_w, bbox_h)
    I_mask = (gt_uint == in_label).astype(int)
    return load_and_pad(I, I_mask, bbox, load_other_masks_fun=load_other_masks_fun, accept_smaller=accept_smaller)

def find_enclosing_crop(bbox_item, h, w, crop_size):
    (bbox_x, bbox_y, bbox_w, bbox_h) = bbox_item
    def calc_bounds_side(bb_start, bb_l, length):
        middle = bb_start + int(0.5*bb_l)
        half_crop_l = int(0.5*crop_size)
        half_crop_r = crop_size - int(0.5*crop_size) #if not div by 2
        space_needed_left = max(half_crop_l-middle, 0)
        available_l = length - 1
        space_needed_right = max((middle + half_crop_r) - available_l, 0)
        start_left = max(middle - half_crop_l, 0) - space_needed_right
        if start_left < 0:
            raise Exception("unable to crop element")
        end_right = min(middle + half_crop_r, available_l) + space_needed_left
        if end_right > length:
            raise Exception("unable to crop element")
        return start_left, end_right
    crop_start_y, crop_end_y = calc_bounds_side(bbox_y, bbox_h, h)
    crop_start_x, crop_end_x = calc_bounds_side(bbox_x, bbox_w, w)
    return crop_start_y, crop_end_y, crop_start_x, crop_end_x


def load_and_pad(I: np.ndarray, I_mask: np.ndarray, bbox, load_other_masks_fun=None, accept_smaller=False):
    crop_size = config['crop_size']
    h,w = I.shape[:2]

    if min(h,w) < crop_size:
        raise Exception("not possible")

    (bbox_x, bbox_y, bbox_w, bbox_h) = bbox
    

    sub = I[bbox_y:(bbox_y + bbox_h),bbox_x:(bbox_x + bbox_w)]
    if load_other_masks_fun is not None:
        other_masks = load_other_masks_fun()
        I_mask = I_mask + other_masks
    tup = (bbox_x, bbox_y, bbox_w, bbox_h)
    ratio = float(crop_size) / max(bbox_h, bbox_w )
    I_bounding = np.zeros((I.shape[0], I.shape[1]))
    I_bounding[bbox_y:(bbox_y + bbox_h),bbox_x:(bbox_x + bbox_w)] = 1.
    if ratio < 1.0:
        sq_x, sq_w, sq_y, sq_h = square_bbox(tup, h, w)
        I_crop = np.zeros((I.shape[0], I.shape[1]))
        I_crop[sq_y:(sq_y + sq_h),sq_x:(sq_x + sq_w)] = 1.

        start_crop_x= max(sq_x - sq_w, 0)
        end_crop_x = min(sq_x + sq_w + sq_w, I.shape[1])
        start_crop_y = max(sq_y - sq_h, 0)
        end_crop_y = min(sq_y + sq_h + sq_h, I.shape[0])

        sub = I[start_crop_y:end_crop_y,start_crop_x:end_crop_x]
        sub_mask = I_mask[start_crop_y:end_crop_y,start_crop_x:end_crop_x]
        sub_bounding = I_bounding[start_crop_y:end_crop_y,start_crop_x:end_crop_x]
        sub_crop = I_crop[start_crop_y:end_crop_y,start_crop_x:end_crop_x]
    
        target_size = (int(sub.shape[0]*ratio),int(sub.shape[1]*ratio))
        sub_small = resize(sub, target_size, anti_aliasing=True)
        sub_mask_small = resize(sub_mask.astype(float), target_size, anti_aliasing=True)
        sub_bounding_small = resize(sub_bounding.astype(float), target_size, anti_aliasing=True)
        sub_crop_small = resize(sub_crop.astype(float), target_size, anti_aliasing=True)
        #correct it
        rmin, rmax, cmin, cmax = get_bounds(sub_crop_small, do_check=False)
        row_length = (rmax + 1) - rmin
        col_length = (cmax + 1) - cmin
        #print(f"row_length: {row_length}, col_length: {col_length}")
        
        sub_crop_small[rmin:(rmin + crop_size), cmin:(cmin + crop_size)] = 1.0
        sub_crop_small[(rmin + crop_size):] = 0.0
        sub_crop_small[:,(cmin + crop_size):] = 0.0
        rmin, rmax, cmin, cmax = get_bounds(sub_crop_small, do_check=False)
        row_length = (rmax + 1) - rmin
        col_length = (cmax + 1) - cmin
        #assert row_length == crop_size and col_length == crop_size
        
        rmin, rmax, cmin, cmax = get_bounds(sub_crop_small, do_check=False)
        
        before_y = max(crop_size - rmin, 0)
        after_y = max(crop_size*3 - (sub_crop_small.shape[0] + before_y), 0)
        before_x = max(crop_size - cmin, 0)
        after_x = max(crop_size*3 - (sub_crop_small.shape[1] + before_x), 0)
        
        to_pad = ((before_y, after_y), (before_x, after_x))
        
        sub_padded = np.pad(sub_small, list(to_pad) + [(0,0)], mode='constant')
        sub_mask_padded = np.pad(sub_mask_small, to_pad, mode='constant')
        sub_bounding_padded = np.pad(sub_bounding_small, to_pad, mode='constant')
        sub_crop_padded = np.pad(sub_crop_small, to_pad, mode='constant')

        length = 3*crop_size
        
        sub_padded = sub_padded[:length,:length]
        assert sub_padded.shape[0] == length and sub_padded.shape[1] == length
        sub_mask_padded = sub_mask_padded[:length,:length]
        sub_bounding_padded = sub_bounding_padded[:length,:length]
        sub_crop_padded = sub_crop_padded[:length,:length]
    else:
        assert accept_smaller
        crop_start_y, crop_end_y, crop_start_x, crop_end_x = find_enclosing_crop(tup, h, w, crop_size)
        assert (crop_end_y - crop_start_y) == crop_size
        assert (crop_end_x - crop_start_x) == crop_size
        temp = I[crop_start_y:(crop_end_y), crop_start_x:(crop_end_x)]
        assert temp.shape[0] == crop_size and temp.shape[1] == crop_size
        start_bound_y = max(crop_start_y - crop_size, 0)
        end_bound_y = min(crop_end_y + crop_size, h)
        start_bound_x = max(crop_start_x - crop_size, 0)
        end_bound_x = min(crop_end_x + crop_size, w)

        do_crop = lambda arr: arr[start_bound_y:(end_bound_y), start_bound_x:(end_bound_x)]
        sub = do_crop(I).astype(float)/255.
        sub_mask = do_crop(I_mask).astype(float)
        sub_bounding = do_crop(I_bounding).astype(float)
        I_crop = np.zeros_like(I_bounding)
        I_crop[crop_start_y:(crop_end_y), crop_start_x:(crop_end_x)] = 1.0
        sub_crop = do_crop(I_crop).astype(float)

        before_y = max(crop_size - crop_start_y, 0)
        after_y = max(crop_size*3 - (sub_crop.shape[0] + before_y), 0)
        before_x = max(crop_size - crop_start_x, 0)
        after_x = max(crop_size*3 - (sub_crop.shape[1] + before_x), 0)
        
        to_pad = ((before_y, after_y), (before_x, after_x))

        sub_padded = np.pad(sub, list(to_pad) + [(0,0)], mode='constant')
        sub_mask_padded = np.pad(sub_mask, to_pad, mode='constant')
        sub_bounding_padded = np.pad(sub_bounding, to_pad, mode='constant')
        sub_crop_padded = np.pad(sub_crop, to_pad, mode='constant')

        length = 3*crop_size
        
        sub_padded = sub_padded[:length,:length]
        assert sub_padded.shape[0] == length and sub_padded.shape[1] == length
        sub_mask_padded = sub_mask_padded[:length,:length]
        sub_bounding_padded = sub_bounding_padded[:length,:length]
        sub_crop_padded = sub_crop_padded[:length,:length]
        
        #idea: 1st: try find square crop
        #2nd: no resizing needed, but pad
        #finished
    import math
    if not math.isclose(sub_mask_padded.max(), 1.0, rel_tol=1e-05, abs_tol=1e-08):
        raise Exception("not enough mask! We are scaling it too small")
    #assert math.isclose(sub_mask_padded.max(), 1.0, rel_tol=1e-01)
    #to coordinates again:
    b_rmin, b_rmax, b_cmin, b_cmax = get_bounds(sub_bounding_padded, do_check=False)
    #correct it...sometimes it get's off by a few pixels if we are scaling a lot
    def correct_b(b_min, b_max):
        leng = (b_max + 1) - b_min
        diff = max(leng - crop_size,0)
        assert diff <= 10 and diff >= 0
        return b_max - diff
    b_rmax = correct_b(b_rmin, b_rmax)
    b_cmax = correct_b(b_cmin, b_cmax)

    temp = sub_bounding_padded[b_rmin:(b_rmax + 1), b_cmin:(b_cmax + 1)]
    assert temp.shape[0] <= crop_size and temp.shape[1] <= crop_size
    bound_coords = ((b_rmin, b_rmax), (b_cmin, b_cmax))

    c_rmin, c_rmax, c_cmin, c_cmax = get_bounds(sub_crop_padded, do_check=False)
    def correct_c(c_min, c_max):
        leng = (c_max + 1) - c_min
        diff = leng - crop_size
        return c_max - diff
    c_rmax = correct_c(c_rmin, c_rmax)
    c_cmax = correct_c(c_cmin, c_cmax)
    temp = sub_crop_padded[c_rmin:(c_rmax + 1), c_cmin:(c_cmax + 1)]
    assert temp.shape[0] == crop_size and temp.shape[1] == crop_size
    crop_coords = ((c_rmin, c_rmax), (c_cmin, c_cmax))

    start_end_coord = ((before_y, crop_size*3-after_y), (before_x, crop_size*3-after_x))
    
    return sub_padded, sub_mask_padded, bound_coords, crop_coords, start_end_coord


def to_torch(img, mask, bound_coord, crop_coord, start_end_coord):
    t_img = torch.from_numpy(img).to(torch.float32).permute(2,0,1).unsqueeze(0)
    t_mask = torch.from_numpy(mask).to(torch.float32).unsqueeze(0)
    t_bound_coord = torch.from_numpy(np.array(bound_coord)).unsqueeze(0).float()
    t_crop_coord = torch.from_numpy(np.array(crop_coord)).unsqueeze(0).float()
    t_start_coord = torch.from_numpy(np.array(start_end_coord)).unsqueeze(0).float()
    return (t_img, t_mask, t_bound_coord, t_crop_coord, t_start_coord)

def expand_data(img, mask, bound_coord, crop_coord, start_end_coord, num_expand):
    size = 3*config['crop_size']
    out_imgs = img.expand(num_expand,3,size,size)
    out_masks = mask.expand(num_expand,size,size)
    out_bounds = bound_coord.expand(num_expand,2,2)
    out_crops = crop_coord.expand(num_expand,2,2)
    out_starts_ends = start_end_coord.expand(num_expand,2,2)
    return out_imgs, out_masks, out_bounds, out_crops, out_starts_ends


def get_centers_helper(rmins, rmaxs, cmins, cmaxs):
    rcenter = (rmins + (rmaxs + 1 - rmins)/2).int()
    ccenter = (cmins + (cmaxs + 1 - cmins)/2).int()
    return torch.stack([rcenter, ccenter], dim=1)

def get_centers(bounds):
    rmins = bounds[:,0,0]
    rmaxs = bounds[:,0,1]
    cmins = bounds[:,1,0]
    cmaxs = bounds[:,1,1]
    return get_centers_helper(rmins, rmaxs, cmins, cmaxs)

def get_zooms(bounds, crops):
    def get_hw(coords):
        rmins = coords[:,0,0]
        rmaxs = coords[:,0,1]
        cmins = coords[:,1,0]
        cmaxs = coords[:,1,1]
        return (rmaxs - rmins), (cmaxs - cmins)
    h_b,w_b = get_hw(bounds)
    h_c,w_c = get_hw(crops)
    r_h = h_b/h_c
    r_w = w_b/w_c
    #assert r_h.max() < 1.0001
    #assert r_w.max() < 1.0001
    return torch.max(r_h, r_w)


class STECeil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.ceil()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

ste_ceil = STECeil.apply

class STEFloor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

ste_floor = STEFloor.apply

def get_bounds_t(coords, do_check=True, quantize=False):
    rmins = coords[:,0,0]
    rmaxs = coords[:,0,1]
    cmins = coords[:,1,0]
    cmaxs = coords[:,1,1]
    if do_check:
        assert torch.allclose((rmaxs - rmins), (cmaxs - cmins))

    if quantize:
        rmins = ste_ceil(rmins)
        rmaxs = ste_floor(rmaxs)
        cmins = ste_ceil(cmins)
        cmaxs = ste_floor(cmaxs)
    return rmins, rmaxs, cmins, cmaxs


def get_zoom_bounds(crops, start_coords):
    h_start, h_end, w_start, w_end = get_bounds_t(start_coords, quantize=True, do_check=False)

    rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops)
    centers = get_centers_helper(rmins, rmaxs, cmins, cmaxs)
    scale_left = (centers[:,0] - rmins)/(centers[:,0] - h_start)
    scale_right = (rmaxs - centers[:,0])/(h_end - centers[:,0])
    scale_top = (centers[:,1] - cmins)/(centers[:,1] - w_start)
    scale_down = (cmaxs - centers[:,1])/(w_end - centers[:,1])
    stacked = torch.stack([scale_left, scale_right, scale_top, scale_down], dim=1)
    return torch.max(stacked, dim=1)[0], centers

def zoom_coords(coords, scales, centers):
    out_coords = torch.zeros_like(coords)
    out_coords[:,:,0] = (centers - scales * (centers - coords[:,:,0]))
    out_coords[:,:,1] = (centers + scales * (coords[:,:,1] - centers))
    return out_coords


def find_height_max_enlosing_rect(crops, start_coords):
    h_start, h_end, w_start, w_end = get_bounds_t(start_coords, quantize=True, do_check=False)
    
    rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops, do_check=False)
    
    space_below = rmins - h_start
    space_above = h_end - (rmaxs)
    space_vertical = torch.min(space_below, space_above)
    space_left = cmins - w_start
    space_right = w_end - (cmaxs)
    space_horizontal = torch.min(space_left, space_right)
    
    inner_heights =(rmaxs - rmins)
    outer_heights = (2*space_vertical + 1) + inner_heights

    inner_widths = (cmaxs - cmins)
    outer_widths = (2*space_horizontal + 1) + inner_widths

    # inner_heights = (h_end +1 - h_start) - 2*space_vertical
    # inner_widths = (w_end +1 - w_start) - 2*space_horizontal
    
    return outer_heights, outer_widths#, inner_heights, inner_widths

def find_height_inclosing_rect_same_center(crops, bounds):
    h_start, h_end, w_start, w_end = get_bounds_t(crops, quantize=True, do_check=False)
    
    rmins, rmaxs, cmins, cmaxs = get_bounds_t(bounds, do_check=False)

    def get_length(i_start,i_end,o_start,o_end):
        left = i_start - o_start
        right = o_end - i_end
        return (o_end + 1 - o_start) - 2*torch.min(left, right)
    
    # space_below = rmins - h_start
    # space_above = h_end - (rmaxs)
    # space_vertical = torch.max(space_below, space_above)
    # space_left = cmins - w_start
    # space_right = w_end - (cmaxs)
    # space_horizontal = torch.max(space_left, space_right)
    
    # inner_heights =(rmaxs - rmins)
    # heights = (2*space_vertical + 1) + inner_heights

    # inner_widths = (cmaxs - cmins)
    # widths = (2*space_horizontal + 1) + inner_widths

    heights = get_length(rmins,rmaxs,h_start,h_end)
    widths = get_length(cmins,cmaxs,w_start,w_end)
    
    return heights, widths
    
import math
def find_max_angle(bounds, crops, start_coords, resize):    
    def calc_max_angles(cube_lengths, outer):
        space_left = (outer.float() - cube_lengths)/2
        # height is:
        # H = C*sin(tetha)*cos(tetha)
        # so
        # (sin * cos)^-1(H/C) =tetha
        max_angle = torch.zeros(outer.shape[0]).to(cube_lengths.device)
        #space_left == 0 => max_angle = 0
        
        ratio_denom = cube_lengths[space_left != 0].float()
        ratio = space_left[space_left != 0] / ratio_denom
        
        calc_max_angle = torch.empty(ratio.shape[0]).to(cube_lengths.device)
        calc_max_angle[ratio >= 1/2.] = 2*np.pi
        calc_max_angle[ratio < 1/2.] = (1/2.*torch.asin(2*ratio[ratio < 1/2.])).abs()
        
        max_angle[space_left != 0] = calc_max_angle
        
        return max_angle

    def calc_max_angle_2(a,b,A):
        max_angle = torch.zeros(A.shape[0], device=A.device)
        denom = torch.sqrt(((a/2)**2) + ((b/2)**2))
        temp1 = (A/2.0)/denom
        temp2 = (a/2.0)/denom
        max_angle[temp1 >= 1.0] = 2*np.pi
        max_angle[temp1 < 1.0] = torch.asin(temp1[temp1 < 1.0])-torch.asin(temp2[temp1 < 1.0])
        return max_angle

    if resize:
        rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops)
        inner_heights =(rmaxs - rmins)
        inner_widths = (cmaxs - cmins)
        assert torch.all(inner_heights == inner_widths)
        enclosing_heights, enclosing_widths = find_height_max_enlosing_rect(crops, start_coords)
        
        max_angles_1 = calc_max_angles(inner_heights, enclosing_heights)
        max_angles_2 = calc_max_angles(inner_widths, enclosing_widths)
        return torch.min(max_angles_1, max_angles_2)      
    else:
        def find_non_resize_angle(inner, outer):
            #inner_heights, inner_widths = find_height_inclosing_rect_same_center(outer, inner)
            enclosing_heights, enclosing_widths = find_height_max_enlosing_rect(inner, outer)
            rmins, rmaxs, cmins, cmaxs = get_bounds_t(inner, do_check=False)
            inner_heights = (rmaxs + 1) - rmins
            inner_widths = (cmaxs + 1) - cmins
            max_angles_1 = calc_max_angle_2(inner_heights, inner_widths, enclosing_heights)
            max_angles_2 = calc_max_angle_2(inner_widths, inner_heights, enclosing_widths)
            return  torch.min(max_angles_1, max_angles_2)

        inner_angle = find_non_resize_angle(bounds, crops)

        outer_angle = find_non_resize_angle(crops, start_coords)

        # #TODO: our picture_bounds (start-coords) could get inside our image!
        # #rmins, rmaxs, cmins, cmaxs = get_bounds_t(bounds, do_check=False)
        # inner_heights, inner_widths = find_height_inclosing_rect_same_center(crops, bounds)
        # # inner_heights =(rmaxs + 1) - rmins
        # # inner_widths = (cmaxs + 1) - cmins
        # enclosing_heights, enclosing_widths = find_height_max_enlosing_rect(bounds, crops)
        
        # # max_angles_1 = calc_max_angles(inner_heights, enclosing_heights)
        # # max_angles_2 = calc_max_angles(inner_widths, enclosing_widths)
        # max_angles_1 = calc_max_angle_2(inner_heights, inner_widths, enclosing_heights)
        # max_angles_2 = calc_max_angle_2(inner_widths, inner_heights, enclosing_widths)
        # inner_angle =  torch.min(max_angles_1, max_angles_2)


        #outer_angle = find_max_angle(bounds, crops, start_coords, True)
        return torch.min(inner_angle, outer_angle)

def rotate(imgs, masks, bounds, crops, start_coords, angles, max_angle, verbose=True, resize=False, disable_check=False):
    if not disable_check:
        with torch.no_grad():
            max_possible_angles = find_max_angle(bounds, crops, start_coords, resize)
            temp = angles[angles.abs()>max_possible_angles]
            rhs = temp.sign()*max_possible_angles[angles.abs()>max_possible_angles]
            angles[angles.abs()>max_possible_angles] = rhs
            temp = angles[angles.abs()>max_angle]
            angles[angles.abs()>max_angle] = temp.sign()*max_angle

    rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops)
    centers = get_centers_helper(rmins, rmaxs, cmins, cmaxs).float()
    centers = torch.flip(centers, dims=(-1,))

    
    real_angles = angles/np.pi * 180.
    if verbose:
        print(f"rotating with {real_angles}")

    real_angles = real_angles.expand(imgs.shape[0])
    centers = centers.expand(centers.shape[0], -1)
    import kornia
    mode: str = 'bilinear'
    padding_mode: str = 'zeros'
    align_corners: bool = True
    rotation_matrix: torch.Tensor = kornia.geometry.transform.affwarp._compute_rotation_matrix(real_angles, centers)

    def rotate_coords(coords, mat):
        rmins, rmaxs, cmins, cmaxs = get_bounds_t(coords, do_check=False)

        # dim [B,2]
        upper_left = torch.stack((rmins, cmins), dim=1)
        upper_right = torch.stack((rmins, cmaxs), dim=1)
        down_left = torch.stack((rmaxs, cmins), dim=1)
        down_right = torch.stack((rmaxs, cmaxs), dim=1)

        # dim [B,4,2]
        rectangle_dims = torch.stack((upper_left, upper_right, down_left, down_right),dim=-2)
        all_dims = torch.flip(rectangle_dims, dims=(-1,))
        # dim [B,4,3]
        all_dims_affine = torch.concat([all_dims, torch.ones((all_dims.shape[0],4,1), device=all_dims.device)], dim=-1)

        # [B,2,3], [B,4,3] -> [B,4,2]
        rotated = torch.einsum("boi, bri -> bro", mat, all_dims_affine)
        rotated_flipped = torch.flip(rotated, dims=(-1,))
        return rotated_flipped

    def rotate_coords_and_fit_rectangle(coords, mat):
        rotated_flipped = rotate_coords(coords, mat)
        start_h = rotated_flipped[:,:,0].min(dim=1)[0]
        end_h = rotated_flipped[:,:,0].max(dim=1)[0]
        bounds_h = torch.stack([start_h, end_h], dim=-1)
        start_w = rotated_flipped[:,:,1].min(dim=1)[0]
        end_w = rotated_flipped[:,:,1].max(dim=1)[0]
        bounds_w = torch.stack([start_w, end_w], dim=-1)

        return torch.stack([bounds_h, bounds_w], dim=-2)

    if resize:
        bounds_crops = rotate_coords_and_fit_rectangle(crops, rotation_matrix[..., :2, :3])
        size_h = bounds_crops[:,0,1] - bounds_crops[:,0,0]
        size_w = bounds_crops[:,1,1] - bounds_crops[:,1,0]
        rotated_size = torch.stack((size_h, size_w), dim=1).max(dim=1)[0]

        scale_factor = float(config['crop_size'])/rotated_size
        if len(scale_factor.shape) == 1:
            scale_factor = scale_factor.unsqueeze(1).repeat(1, 2)
            #scale_factor = scale_factor.repeat(1, 2)
        scaling_matrix: torch.Tensor = kornia.geometry.transform.affwarp._compute_scaling_matrix(scale_factor, centers)
        ones = torch.tensor([0,0,1], device=rotation_matrix.device).view(1,1,3).expand((scaling_matrix.shape[0],1,3))
        temp = torch.concat([rotation_matrix[..., :2, :3], ones], dim=1)
        operation = torch.bmm(scaling_matrix[..., :2, :3], temp)

        if True:
            bounds_crops = rotate_coords_and_fit_rectangle(crops, operation)
            size_h = bounds_crops[:,0,1] - bounds_crops[:,0,0]
            size_w = bounds_crops[:,1,1] - bounds_crops[:,1,0]
            rotated_size = torch.stack((size_h, size_w), dim=1).max(dim=1)[0]
            assert torch.allclose(rotated_size, torch.tensor(float(config['crop_size'])))

        rotate_imgs = t.affine(imgs, operation, mode, padding_mode, align_corners)
        rotate_masks = t.affine(masks.unsqueeze(1), operation, mode, padding_mode, align_corners).squeeze(1)

        rotated_bounds = rotate_coords_and_fit_rectangle(bounds, operation)
        #rotated_start_coords = transform_coords(start_coords)

        return rotate_imgs, rotate_masks, rotated_bounds, crops, None
    else:
        #plot_debug_2(rotation_matrix[..., :2, :3])
        bounds_bounds = rotate_coords_and_fit_rectangle(crops, rotation_matrix[..., :2, :3])
        rotate_imgs = t.affine(imgs, rotation_matrix[..., :2, :3], mode, padding_mode, align_corners)
        rotate_masks = t.affine(masks.unsqueeze(1), rotation_matrix[..., :2, :3], mode, padding_mode, align_corners).squeeze(1)
        return rotate_imgs, rotate_masks, bounds_bounds, crops, None


def random_rotation(imgs, masks, bounds, crops, start_coords, max_angle=45./180.*np.pi):
    angles = find_max_angle(crops, start_coords)
    random = -angles + (2*angles * torch.rand(imgs.shape[0]))
    return rotate(imgs, masks, bounds, crops, start_coords, random, max_angle), random

def calculate_translate_bounds(crops, bounds, start_coords):
    i_rmins, i_rmaxs, i_cmins, i_cmaxs = get_bounds_t(bounds, do_check=False)
    
    rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops)
    
    space_below = i_rmins - rmins
    space_above = (rmaxs - 1) - (i_rmaxs - 1)
    space_left = i_cmins - cmins
    space_right = (cmaxs - 1) - (i_cmaxs - 1)

    h_start, h_end, w_start, w_end = get_bounds_t(start_coords, quantize=True, do_check=False)
    
    image_space_above = h_end - rmaxs
    image_space_below = rmins - h_start
    image_space_left = cmins - w_start
    image_space_right = w_end - cmaxs
    
    
    
    move_max_below = torch.min(image_space_above, space_below)
    move_max_above = torch.min(image_space_below, space_above)
    move_max_left = torch.min(image_space_left, space_right)
    move_max_right = torch.min(image_space_right, space_left)
    
    #todo we might shift too much and
    
    return move_max_below, move_max_above, move_max_left, move_max_right

def translate_xy(imgs, masks, bounds, crops, start_coords, trans_x, trans_y, verbose=True):
    with torch.no_grad():
        t_bounds = calculate_translate_bounds(crops, bounds, start_coords)
        move_max_below, move_max_above, move_max_left, move_max_right = t_bounds
        dtype = trans_x.dtype
        
        move_max_below = move_max_below.to(dtype)
        move_max_above = move_max_above.to(dtype)
        move_max_left = move_max_left.to(dtype)
        move_max_right = move_max_right.to(dtype)

        trans_x[trans_x < -move_max_above] = -move_max_above[trans_x < -move_max_above]
        trans_x[trans_x > move_max_below] = move_max_below[trans_x > move_max_below]

        # trans_y[trans_y < -move_max_right] = -move_max_right[trans_y < -move_max_right]
        # trans_y[trans_y > move_max_left] = move_max_left[trans_y > move_max_left]
        
        # trans_x[trans_x < -move_max_below] = -move_max_below[trans_x < -move_max_below]
        # trans_x[trans_x > move_max_above] = move_max_above[trans_x > move_max_above]

        trans_y[trans_y < -move_max_left] = -move_max_left[trans_y < -move_max_left]
        trans_y[trans_y > move_max_right] = move_max_right[trans_y > move_max_right]
    
    if verbose:
        print(f"real translating with {trans_x, trans_y}")
    t_vecs = torch.stack([trans_y, trans_x], dim=1)
    t_vecs = (-1)*t_vecs
    do_it = lambda i: t.translate(i, t_vecs)
    out_imgs = do_it(imgs)
    out_masks = do_it(masks.unsqueeze(1)).squeeze(1)

    len_available_b = 3*config['crop_size']

    out_bounds = bounds.clone()
    out_bounds[:,0] = (out_bounds[:,0] - trans_x.unsqueeze(1)).clamp(0,len_available_b)
    out_bounds[:,1] = (out_bounds[:,1] - trans_y.unsqueeze(1)).clamp(0,len_available_b)
    
    start_ct = start_coords.clone()
    start_ct[:,0] = (start_ct[:,0] - trans_x.unsqueeze(1)).clamp(0,len_available_b)
    start_ct[:,1] = (start_ct[:,1] - trans_y.unsqueeze(1)).clamp(0,len_available_b)
    
    return out_imgs, out_masks, out_bounds, crops, start_ct
    
    #out_crops = do_it(crops)
    #return imgs, masks, bounds, out_crops, start_coords

def random_translate(imgs, masks, bounds, crops, start_coords):
    t_bounds = calculate_translate_bounds(crops, bounds, start_coords)
    move_max_below, move_max_above, move_max_left, move_max_right = t_bounds
    print(move_max_below, move_max_above, move_max_left, move_max_right)
    t_x = -move_max_below + (move_max_below + move_max_above) * torch.rand(imgs.shape[0])
    t_y = -move_max_left + (move_max_left + move_max_right) * torch.rand(imgs.shape[0])
    print(f"translating with {t_x, t_y}")
    return translate_xy(imgs, masks, bounds, crops, start_coords, t_x, t_y), (t_x,t_y)

def crop_batches(imgs: torch.Tensor, masks: torch.Tensor, bounds: torch.Tensor,
    crops: torch.Tensor, start_coords: torch.Tensor):
    start_y = crops[:,0,0]
    start_x = crops[:,1,0]
    trans_y = config['crop_size']-start_y
    trans_x = config['crop_size']-start_x
    t_vecs = torch.stack([trans_x, trans_y], dim=1)
    do_it = lambda i: t.translate(i, t_vecs)
    if not (t_vecs == 0).all().item():
        out_imgs = do_it(imgs)
        out_masks = do_it(masks.unsqueeze(1)).squeeze(1)
    else:
        out_imgs = imgs
        out_masks = masks
    start = config['crop_size']
    end = config['crop_size'] + config['crop_size']
    cropped_imgs = out_imgs[:,:,start:(end),start:(end)]
    cropped_masks = out_masks[:,start:(end),start:(end)]
    max_m = cropped_masks.amax(dim=(1, 2))

    bounds_res = bounds.clone()
    bounds_res[:,0] = bounds_res[:,0] - start_y.unsqueeze(-1)
    bounds_res[:,1] = bounds_res[:,1] - start_x.unsqueeze(-1)

    crops_res = torch.zeros_like(bounds_res)
    crops_res[:,:,1] = config['crop_size']

    start_coords = torch.zeros_like(bounds_res)
    start_coords[:,:,1] = config['crop_size']

    return cropped_imgs, cropped_masks, bounds_res, crops_res, start_coords

def rescale_cropped(imgs: torch.Tensor, masks: torch.Tensor, bounds: torch.Tensor,
    crops: torch.Tensor, start_coords: torch.Tensor):
    if config['crop_size'] != config['target_size']:
        scale = float(config['target_size'])/float(config['crop_size'])
        scale_tensor = torch.tensor([scale], device=imgs.device).expand(imgs.shape[0]).unsqueeze(-1)
        scaled_cropped = t.scale(imgs, scale_tensor)
        scaled_cropped_masks = t.scale(masks.unsqueeze(1), scale_tensor).squeeze(1)
        start = int((config['crop_size'] != config['target_size'])/2)
        end = start + config['target_size']
        s_c = scaled_cropped[:,:,start:(end),start:(end)]
        s_m = scaled_cropped_masks[:,start:(end),start:(end)]

        rmins, rmaxs, cmins, cmaxs = get_bounds_t(crops)
        centers = get_centers_helper(rmins, rmaxs, cmins, cmaxs)

        def rescale_coords(unscaled_coords):
            coords = zoom_coords(unscaled_coords, scale_tensor, centers)
            coords[:,0] = coords[:,0] - start
            coords[:,1] = coords[:,1] - start
            return coords

        r_bounds = rescale_coords(bounds)
        r_crops = rescale_coords(crops)
        r_start_coords = rescale_coords(start_coords)

        assert torch.allclose(r_crops[:,:,0], torch.zeros_like(r_crops))
        assert torch.allclose(r_crops[:,:,1], torch.zeros_like(r_crops) + config['target_size'])


        return s_c, s_m, r_bounds, r_crops, r_start_coords
    else:
        return imgs, masks, bounds, crops, start_coords