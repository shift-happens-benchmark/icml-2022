import glob
import os
import re

import numpy as np
import pandas as pd
import torch


def build_map_dict():
    """builds a dictionary for Imagenet classes: index to class name.

    Returns:
        dict: the dictionary
    """
    url = "https://domainadaptation.org/selflearning/imagenet_classes.csv"
    df = pd.read_csv(url, sep = '\001', header=None, names=["class_index", "class_name"])
    map_dict = {}
    for _, row in df.iterrows():
        map_dict[row["class_index"]] = row["class_name"]
    return map_dict


def get_imagenet_visda_mapping(visda_dir, map_dict):
    """builds a dictionary that maps from VISDA classes to Imagenet classes.

    Args:
        visda_dir (str): VISDA dataset path. It's used to get the classes names.
        map_dict (dict): The Imagenet dictionary: index to class names.

    Returns:
        matching_names (dict): mapping of class names
        matching_labels (dict): mapping of labels
    """
    matching_names = dict()
    matching_labels = dict()
    map_dict_visda = dict()

    visda = os.listdir(visda_dir)
    for label, item in enumerate(sorted(visda)):
        map_dict_visda[item] = label
        item_split = item.split("_")
        for ii in item_split:
            for j in map_dict:
                if re.search(r"\b" + ii + r"\b", map_dict[j]):
                    if item not in matching_names:
                        matching_names[item] = list()
                        matching_labels[str(label)] = list()

                    matching_names[item].append([map_dict[j]])
                    matching_labels[str(label)].append(j)

    matching_names, matching_labels = clean_dataset(
        matching_names, matching_labels, map_dict_visda, map_dict
    )

    return matching_names, matching_labels


def create_symlinks_and_get_imagenet_visda_mapping(
    visda_location, symlinks_location, map_dict
):
    """Creates symlinks between Imagenet class names to visda folders with images.
    
    Args:
        visda_location (str): VISDA dataset path.
        symlinks_location (str): The second parameter.
        map_dict (dict): The Imagenet dictionary: index to class names.

    Returns:
        dict: the dictionary that maps visda classes to imagenet classes.
    """

    # initial mapping and cleaning
    matching_names, matching_labels = get_imagenet_visda_mapping(
        visda_location, map_dict
    )

    # some classes are ambiguous
    ambiguous_matching = get_ambiguous_classes(matching_names)

    # create symlinks for all valid classes
    symlinks_location = os.path.abspath(symlinks_location)
    if not os.path.exists(symlinks_location):
        os.makedirs(symlinks_location)
    else:
        print("Path ", symlinks_location, " exists, skipping.")
    for folder in matching_names.keys():
        target_folder_class = os.path.join(
            symlinks_location, ambiguous_matching[folder]
        )
        if not os.path.exists(target_folder_class):
            os.makedirs(target_folder_class)
        try:
            allFiles_path_jpg = os.path.join(visda_location, folder, "*.jpg")
            allFiles_path_png = os.path.join(visda_location, folder, "*.png")
            allFiles_jpg = glob.glob(allFiles_path_jpg)
            allFiles_png = glob.glob(allFiles_path_png)
            allFiles = allFiles_jpg + allFiles_png
            for file in allFiles:
                newFile = os.path.join(target_folder_class, os.path.normpath(os.path.basename(file)))
                file = os.path.abspath(file)
                os.symlink(file, newFile)
        except FileExistsError:
            pass

    # final mapping and cleaning with the symlinks
    matching_names, matching_labels = get_imagenet_visda_mapping(
        symlinks_location, map_dict
    )

    mapping_vector = torch.zeros((1000))
    mapping_vector -= 1
    mapping_vector_counts = dict()
    for i in range(1000):
        if i not in mapping_vector_counts.keys():
            mapping_vector_counts[i] = list()
        for j in matching_labels:
            if i in matching_labels[j]:
                mapping_vector[i] = int(j)
                mapping_vector_counts[i].append(j)

    # if classes are mapped to more than one class, we want to know about it:
    for i in mapping_vector_counts.keys():
        if len(mapping_vector_counts[i]) > 1:
            print(
                map_dict[i], i, "is mapped to visda classes: ", mapping_vector_counts[i]
            )

    return mapping_vector


def clean_dataset(matching_names, matching_labels, map_dict_visda, map_dict):
    """It removes the classes that don't have a mapping between the 2 datasets.
    And it groups together imagenet classes that have a single mapping in VISDA.
    Example: ("wild boar", "warthog", "piggy bank") will map to ("pig").

    Args:
        matching_names (dict): mapping of class names: VISDA -> Imagenet.
        matching_labels (dict): mapping of class labels: VISDA -> Imagenet.
        map_dict_visda (dict): mapping of VISDA labels to class names.
        map_dict (dict): mapping of Imagenet labels to class names.

    Returns:
        matching_names (dict): the cleaned matching_names.
        matching_labels (dict): the cleaned matching_labels.
    """

    # delete labels completely
    del_list = [
        "mouse",
        "fish",
        "light_bulb",
        "leaf",
        "face",
        "wine_glass",
        "hockey_stick",
        "star",
        "see_saw",
        "pencil",
        "grass",
        "fire_hydrant",
        "brain",
        "apple",
        "river",
        "rhinoceros",
        "power_outlet",
        "rain",
        "pool",
        "picture_frame",
        "paper_clip",
        "palm_tree",
        "paint_can",
        "mouth",
        "The_Great_Wall_of_China",
        "garden",
        "garden_hose",
        "hand",
        "house_plant",
        "jacket",
        "tree",
        "sun",
        "smiley_face",
        "beach",
        "diving_board",
        "mountain",
    ]
    for item in del_list:
        try:
            del matching_names[item]
            del matching_labels[str(map_dict_visda[item])]
        except:
            pass
    # delete some imagenet labels

    del matching_names["cat"][5:]
    del matching_labels[str(map_dict_visda["cat"])][5:]

    del matching_names["dog"][-1]
    del matching_labels[str(map_dict_visda["dog"])][-1]

    del matching_names["pig"][0]
    del matching_labels[str(map_dict_visda["pig"])][0]

    del matching_names["bear"][-1]
    del matching_labels[str(map_dict_visda["bear"])][-1]

    del matching_names["horse"][0]
    del matching_labels[str(map_dict_visda["horse"])][0]

    del matching_names["hot_air_balloon"][0:2]
    del matching_labels[str(map_dict_visda["hot_air_balloon"])][0:2]

    del matching_names["hot_dog"][2:15]
    del matching_labels[str(map_dict_visda["hot_dog"])][2:15]

    del matching_names["house"][0]
    del matching_labels[str(map_dict_visda["house"])][0]

    del matching_names["ice_cream"][0]
    del matching_labels[str(map_dict_visda["ice_cream"])][0]

    del matching_names["kangaroo"][1]
    del matching_labels[str(map_dict_visda["kangaroo"])][1]

    del matching_names["washing_machine"][1:-1]
    del matching_labels[str(map_dict_visda["washing_machine"])][1:-1]

    del matching_names["traffic_light"][1:-1]
    del matching_labels[str(map_dict_visda["traffic_light"])][1:-1]

    del matching_names["table"][-1]
    del matching_labels[str(map_dict_visda["table"])][-1]

    del matching_names["stop_sign"][0]
    del matching_labels[str(map_dict_visda["stop_sign"])][0]

    del matching_names["spider"][-2]
    del matching_labels[str(map_dict_visda["spider"])][-2]

    del matching_names["snake"][-2:]
    del matching_labels[str(map_dict_visda["snake"])][-2:]

    del matching_names["sleeping_bag"][1]
    del matching_labels[str(map_dict_visda["sleeping_bag"])][1]

    del matching_names["sleeping_bag"][1]  # not a bug that this comes twice
    del matching_labels[str(map_dict_visda["sleeping_bag"])][1]

    del matching_names["sheep"][0]
    del matching_labels[str(map_dict_visda["sheep"])][0]

    del matching_names["sea_turtle"][:-4]
    del matching_labels[str(map_dict_visda["sea_turtle"])][:-4]

    del matching_names["squirrel"][1]
    del matching_labels[str(map_dict_visda["squirrel"])][1]

    del matching_names["lion"][0]
    del matching_labels[str(map_dict_visda["lion"])][0]

    del matching_names["bee"][0]
    del matching_labels[str(map_dict_visda["bee"])][0]

    del matching_names["bee"][-1]
    del matching_labels[str(map_dict_visda["bee"])][-1]

    del matching_names["soccer_ball"][1:]
    del matching_labels[str(map_dict_visda["soccer_ball"])][1:]

    del matching_names["tractor"][1]
    del matching_labels[str(map_dict_visda["tractor"])][1]

    del matching_names["oven"][-1]
    del matching_labels[str(map_dict_visda["oven"])][-1]

    del matching_names["piano"][0]
    del matching_labels[str(map_dict_visda["piano"])][0]

    del matching_names["barn"][0]
    del matching_labels[str(map_dict_visda["barn"])][0]

    del matching_names["tiger"][0:2]
    del matching_labels[str(map_dict_visda["tiger"])][0:2]

    del matching_names["tiger"][-1]
    del matching_labels[str(map_dict_visda["tiger"])][-1]

    del matching_names["monkey"][0]
    del matching_labels[str(map_dict_visda["monkey"])][0]

    del matching_names["bear"][-2:]
    del matching_labels[str(map_dict_visda["bear"])][-2:]

    del matching_names["car"][2]
    del matching_labels[str(map_dict_visda["car"])][2]

    del matching_names["car"][-1]
    del matching_labels[str(map_dict_visda["car"])][-1]

    # add items:
    matching_names["airplane"] = [
        ["warplane, military plane"],
        ["airliner"],
        ["airship, dirigible"],
    ]
    matching_labels[str(map_dict_visda["airplane"])] = [895, 404, 405]

    matching_names["t-shirt"] = ["jersey, T-shirt, tee shirt"]
    matching_labels[str(map_dict_visda["t-shirt"])] = [610]

    matching_names["teddy-bear"] = ["teddy, teddy bear"]
    matching_labels[str(map_dict_visda["teddy-bear"])] = [850]

    matching_names["bicycle"].append(["mountain bike, all-terrain bike, off-roader"])
    matching_labels[str(map_dict_visda["bicycle"])].extend([671])

    matching_names["bus"].append(["trolleybus, trolley coach, trackless trolley"])
    matching_labels[str(map_dict_visda["bus"])].extend([874])

    matching_names["bus"].append(["minibus"])
    matching_labels[str(map_dict_visda["bus"])].extend([654])

    matching_names["frog"].append(["bullfrog, Rana catesbeiana"])
    matching_labels[str(map_dict_visda["frog"])].extend([30])

    matching_names["rabbit"].append(["hare"])
    matching_labels[str(map_dict_visda["rabbit"])].extend([331])

    matching_names["sea_turtle"].append(["terrapin"])
    matching_labels[str(map_dict_visda["sea_turtle"])].extend([36])

    matching_names["whale"].append(["dugong, Dugong dugon"])
    matching_labels[str(map_dict_visda["whale"])].extend([149])

    matching_names["pig"].append(["wild boar, boar, Sus scrofa"])
    matching_labels[str(map_dict_visda["pig"])].extend([342])

    matching_names["pig"].append(["warthog"])
    matching_labels[str(map_dict_visda["pig"])].extend([343])

    matching_names["pig"].append(["piggy bank, penny bank"])
    matching_labels[str(map_dict_visda["pig"])].extend([719])

    matching_names["car"].append(
        ["police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria"]
    )
    matching_labels[str(map_dict_visda["car"])].extend([734])

    # add dogs to dog label:
    matching_labels[str(map_dict_visda["dog"])].extend(list(np.arange(151, 269)))
    for i in np.arange(151, 269):
        if map_dict[i] not in matching_names["dog"]:
            matching_names["dog"].append([map_dict[i]])

    # add more butterflies:
    matching_labels[str(map_dict_visda["butterfly"])].extend(list(np.arange(320, 322)))
    for i in np.arange(320, 322):
        if map_dict[i] not in matching_names["butterfly"]:
            matching_names["butterfly"].append([map_dict[i]])

    # add more mosquitos:
    matching_labels[str(map_dict_visda["mosquito"])].extend(list(np.arange(318, 320)))
    for i in np.arange(318, 320):
        if map_dict[i] not in matching_names["mosquito"]:
            matching_names["mosquito"].append([map_dict[i]])

    # add more monkeys:
    matching_labels[str(map_dict_visda["monkey"])].extend(list(np.arange(365, 385)))
    for i in np.arange(365, 385):
        if map_dict[i] not in matching_names["monkey"]:
            matching_names["monkey"].append([map_dict[i]])

    # add more snakes:
    matching_labels[str(map_dict_visda["snake"])].extend(list(np.arange(52, 69)))
    for i in np.arange(52, 69):
        if map_dict[i] not in matching_names["snake"]:
            matching_names["snake"].append([map_dict[i]])

    # add more spiders:
    matching_labels[str(map_dict_visda["spider"])].extend(list(np.arange(72, 79)))
    for i in np.arange(72, 79):
        if map_dict[i] not in matching_names["spider"]:
            matching_names["spider"].append([map_dict[i]])

    matching_names["spider"].append(["harvestman, daddy longlegs, Phalangium opilio"])
    matching_labels[str(map_dict_visda["spider"])].extend([70])

    # add more birds:
    matching_labels[str(map_dict_visda["bird"])].extend(list(np.arange(80, 101)))
    for i in np.arange(80, 101):
        if map_dict[i] not in matching_names["bird"]:
            matching_names["bird"].append([map_dict[i]])

    # add more birds:
    matching_labels[str(map_dict_visda["bird"])].extend(list(np.arange(7, 24)))
    for i in np.arange(7, 24):
        if map_dict[i] not in matching_names["bird"]:
            matching_names["bird"].append([map_dict[i]])

    # remove dublicates from labels:
    for item in matching_labels:
        tmp = set(matching_labels[item])
        matching_labels[item] = list(tmp)

    return matching_names, matching_labels


def map_imagenet_class_to_visda_class(pred_label, mapping_vector):
    """Get the equivalent VISDA class label for an input imagenet class label
    
    Args:
        pred_label (long): Imagenet class label.
        mapping_vector (dict): mapping of classes: VISDA -> Imagenet.

    Returns:
        pred_label_visda_tensor (long): VISDA class label.
    """
    pred_label_visda_tensor = mapping_vector[pred_label].long()
    return pred_label_visda_tensor


def map_visda_class_to_imagenet_class(pred_label, mapping_vector):
    """Get the equivalent iamgenet class label for an input VISDA class label
    
    Args:
        pred_label (long): VISDA class label.
        mapping_vector (dict): mapping of classes: VISDA -> Imagenet.

    Returns:
        pred_label_visda_tensor (long): Imagenet class label.
    """
    pred_label_visda_tensor = mapping_vector[pred_label].long()
    return pred_label_visda_tensor


def get_ambiguous_classes(matching_names):
    """Renames ambiguous classes to their VISDA class names;
    
    Args:
        matching_names (dict): Imagenet to VISDA class names.

    Returns:
        matching_names (dict): Imagenet to VISDA class names.
    """    
    ambiguous_matching = {}

    ambiguous_matching["telephone"] = "telephone"
    ambiguous_matching["cell_phone"] = "telephone"

    ambiguous_matching["fan"] = "fan"
    ambiguous_matching["ceiling_fan"] = "fan"

    ambiguous_matching["clock"] = "clock"
    ambiguous_matching["alarm_clock"] = "clock"

    ambiguous_matching["bathtub"] = "bathtub"
    ambiguous_matching["hot_tub"] = "bathtub"

    ambiguous_matching["baseball"] = "baseball"
    ambiguous_matching["baseball_bat"] = "baseball"

    ambiguous_matching["bed"] = "bed"
    ambiguous_matching["couch"] = "bed"

    ambiguous_matching["car"] = "car"
    ambiguous_matching["police_car"] = "car"

    ambiguous_matching["coffee_cup"] = "cup"
    ambiguous_matching["cup"] = "cup"
    ambiguous_matching["mug"] = "cup"

    ambiguous_matching["computer"] = "computer"
    ambiguous_matching["keyboard"] = "computer"
    ambiguous_matching["laptop"] = "computer"

    ambiguous_matching["ice_cream"] = "ice_cream"
    ambiguous_matching["lollipop"] = "ice_cream"
    ambiguous_matching["popsicle"] = "ice_cream"

    ambiguous_matching["bus"] = "bus"
    ambiguous_matching["school_bus"] = "bus"

    ambiguous_matching["truck"] = "truck"
    ambiguous_matching["pickup_truck"] = "truck"
    ambiguous_matching["van"] = "truck"
    ambiguous_matching["firetruck"] = "truck"

    ambiguous_matching["bird"] = "bird"
    ambiguous_matching["swan"] = "bird"

    for key in matching_names.keys():
        if key not in ambiguous_matching:
            ambiguous_matching[key] = key

    return ambiguous_matching
