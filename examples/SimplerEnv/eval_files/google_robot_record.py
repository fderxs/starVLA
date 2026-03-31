import os
import time
from tqdm import tqdm
import argparse

TASKS_DICT = {
    "pick_coke_can_vm": {
        "scene": ["google_pick_coke_can_1_v4"],
        "env": ["GraspSingleOpenedCokeCanInScene-v0"],
        "extra": "urdf_version",
        "total": 300,
        "sub_total": 12,
    },
    "pick_coke_can_va": {
        "scene": ["google_pick_coke_can_1_v4", "google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2", "Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"],
        "env": ["GraspSingleOpenedCokeCanInScene-v0", "GraspSingleOpenedCokeCanDistractorInScene-v0", "GraspSingleOpenedCokeCanAltGoogleCameraInScene-v0", "GraspSingleOpenedCokeCanAltGoogleCamera2InScene-v0"],
        "total": 825,
        "sub_total": 33,
    },
    "move_near_vm": {
        "scene": ["google_pick_coke_can_1_v4", "google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2", "Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"],
        "env": ["MoveNearGoogleBakedTexInScene-v0"],
        "total": 240,
        "sub_total": 24,
    },
    "move_near_va": {
        "scene": ["google_pick_coke_can_1_v4", "google_pick_coke_can_1_v4_alt_background", "google_pick_coke_can_1_v4_alt_background_2", "Baked_sc1_staging_objaverse_cabinet1_h870", "Baked_sc1_staging_objaverse_cabinet2_h870"],
        "env": ["MoveNearGoogleInScene-v0", "MoveNearAltGoogleCameraInScene-v0","MoveNearAltGoogleCamera2InScene-v0"],
        "total": 600,
        "sub_total": 60,
    },
    "drawer_vm": {
        "scene": ["dummy_drawer"],
        "env": ["OpenTopDrawerCustomInScene-v0", "OpenMiddleDrawerCustomInScene-v0", "OpenBottomDrawerCustomInScene-v0", "CloseTopDrawerCustomInScene-v0", "CloseMiddleDrawerCustomInScene-v0", "CloseBottomDrawerCustomInScene-v0"],
        "total": 216,
        "sub_total": 24,
    },
    "drawer_va": {
        "scene": ["frl_apartment_stage_simple", "modern_bedroom_no_roof", "modern_office_no_roof"],
        "env": ["OpenTopDrawerCustomInScene-v0", "OpenMiddleDrawerCustomInScene-v0", "OpenBottomDrawerCustomInScene-v0", "CloseTopDrawerCustomInScene-v0", "CloseMiddleDrawerCustomInScene-v0", "CloseBottomDrawerCustomInScene-v0"],
        "total": 378,
        "sub_total": 42,
    },
    "put_in_drawer_vm": {
        "scene": ["dummy_drawer"],
        "env": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
        "total": 108,
        "sub_total": 12,
    },
    "put_in_drawer_va": {
        "scene": ["frl_apartment_stage_simple", "modern_bedroom_no_roof", "modern_office_no_roof"],
        "env": ["PlaceIntoClosedTopDrawerCustomInScene-v0"],
        "total": 189,
        "sub_total": 21,
    }
}

existing_files = set()

def record(file_path):
    
    for task, task_info in TASKS_DICT.items():
        if any([scene in file_path for scene in task_info["scene"]]) and any([env in file_path for env in task_info["env"]]) and (not "extra" in task_info or task_info["extra"] in file_path):
            task_info["cnt"] += 1
            if "success" in file_path.split("/")[-1]:
                task_info["scc"] += 1
            if task_info["bar"] is not None:
                task_info["bar"].update(1)
                task_info["bar"].set_description(f"{task:<20}scc< {task_info['scc'] / (task_info['cnt'])*100:04.1f}% >")
            break
    else:
        raise ValueError(f"Unknown task: {file_path}")

def search_new_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                if full_path not in existing_files:
                    existing_files.add(full_path)
                    record(full_path)

def task_init():
    for task in TASKS_DICT.keys():
        TASKS_DICT[task]["cnt"] = 0
        TASKS_DICT[task]["scc"] = 0
        TASKS_DICT[task]["bar"] = None
        TASKS_DICT[task]["scc_rate"] = 0

    search_new_files(args.path)
    for task, task_info in TASKS_DICT.items():
        TASKS_DICT[task]["bar"] = tqdm(total=TASKS_DICT[task]["total"], initial=TASKS_DICT[task]["cnt"], ncols=120, desc=f"{task:<20}scc< {task_info['scc'] / (task_info['cnt']+1e-5)*100:04.1f}% >")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--path", type=str, default="results/google_robot/configs_origin+fractal+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--libero--90000_chkpt")
    args.add_argument("--steps", type=str, default=None)
    args.add_argument("-s", "--sub", action="store_true")

    if args.sub:
        for task, task_info in TASKS_DICT.items():
            task_info["total"] = task_info["sub_total"]
    
    args = args.parse_args()
    if args.steps is not None:
        args.path = args.path.replace("90000", args.steps)

    print(f"Start recording from {args.path}")
    task_init()

    try:
        while True:
            time.sleep(1)
            search_new_files(args.path)
            
    except KeyboardInterrupt:
        for task_info in TASKS_DICT.values(): task_info["bar"].close()
        
        tab = lambda tk: '\t'*2 if len(tk)>len('drawer_va') else '\t'*3

        print("\n\n++++++++++++++++++++++TOTAL+++++++++++++++++++++++++")
        for task, task_info in TASKS_DICT.items():
            print(f"{task}_tot: {tab(task)}{task_info['cnt']} ({task_info['cnt'] / task_info['total']*100:.1f}% finished)")
        
        print("\n\n++++++++++++++++++++SUCCESS RATE++++++++++++++++++++")
        for task, task_info in TASKS_DICT.items():
            task_info["scc_rate"] = task_info['scc'] / task_info['cnt']*100
            print(f"{task}_scc: {tab(task)}{task_info['scc_rate']:04.1f}%")

        print("\n\n+++++++++++++++AVERAGE SUCCESS RATE+++++++++++++++++")
        print(f"average_vm_success_rate:\t{sum([task_info['scc_rate'] for task, task_info in TASKS_DICT.items() if 'vm' in task])/4:04.1f}%")
        print(f"average_va_success_rate:\t{sum([task_info['scc_rate'] for task, task_info in TASKS_DICT.items() if 'va' in task])/4:04.1f}%")