"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # (space saving option) extract 84x84 image observations with compression and without 
    # extracting next obs (not needed for pure imitation learning algos)
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 \
        --compress --exclude-next-obs

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""

import os
import json
import h5py
import argparse
import numpy as np
from copy import deepcopy

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from action_extractor.utils.robosuite_data_processing_utils import replace_all_lights, recolor_gripper, recolor_robot, insert_camera_info

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
    
def exclude_cameras_from_obs(traj, camera_names, store_voxel, store_point_cloud):
    if len(camera_names) > 0:
        for cam in camera_names:
            del traj['obs'][f"{cam}_image"]
            del traj['obs'][f"{cam}_depth"]
            del traj['obs'][f"{cam}_rgbd"]
    # if not store_voxel:
    #     del traj['obs']['voxels']
    # if not store_point_cloud:
    #     del traj['obs']['pointcloud_points']
    #     del traj['obs']['pointcloud_colors']


def visualize_voxel(traj):
    
    np_voxels = traj['obs']['voxels'][0]
    #occupancy = traj['obs']['voxels'][0][0,:,:,:]
    #indices = np.argwhere(occupancy == 1)[0]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    indices = np.argwhere(np_voxels[0] != 0)
    colors = np_voxels[1:, indices[:, 0], indices[:, 1], indices[:, 2]].T

    ax.scatter(indices[:, 0], indices[:, 1], indices[:, 2], c=colors/255., marker='s')

    # Set labels and show the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_zlim(0, 64)  

    plt.show()

def extract_trajectory(
    env_meta,
    args, 
    camera_names,
    initial_state, 
    states, 
    actions,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    done_mode = args.done_mode
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    insert_camera_info(initial_state)
    obs = env.reset_to(initial_state)

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):

        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[-1])
        else:
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states" : states[t]})

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def worker(x):
    env_meta, args, camera_names, initial_state, states, actions = x
    traj = extract_trajectory(
        env_meta=env_meta,
        args=args,
        camera_names=camera_names,
        initial_state=initial_state, 
        states=states, 
        actions=actions,
    )
    return traj

def preprocess_depth(traj, cam_name, depth_minmax):
    # depths.shape = (N, H, W, C)
    minmax = depth_minmax[cam_name]
    minmax_range = minmax[1] - minmax[0]
    ndepths =(np.clip(traj['obs'][f'{cam_name}_depth'], minmax[0], minmax[1]) - minmax[0]) / minmax_range * 255
    return ndepths.astype(np.uint8)

def add_rgbd_obs(traj, camera_names, depth_minmax):
    for cam_name in camera_names:
        traj['obs'][f'{cam_name}_rgbd'] = np.concatenate([traj['obs'][f'{cam_name}_image'],
                                                          preprocess_depth(traj, cam_name, depth_minmax)],
                                                            axis=3)
        traj['next_obs'][f'{cam_name}_rgbd'] = np.concatenate([traj['next_obs'][f'{cam_name}_image'],
                                                            preprocess_depth(traj, cam_name, depth_minmax)],
                                                            axis=3)
        del traj['obs'][f'{cam_name}_depth']
        del traj['next_obs'][f'{cam_name}_depth']
    return traj

def dataset_states_to_obs(args):
    store_voxel = args.store_voxel
    store_point_cloud = args.store_point_cloud
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    # original_gripper = env_meta['env_kwargs']['gripper_types']
    env_meta['env_kwargs']['gripper_types'] = 'PandaGripper'
    camera_names = ['agentview', 'frontview', 'fronttableview', 'robot0_eye_in_hand']
    additional_camera_for_voxel = [] if store_voxel or store_point_cloud else []
    camera_names = camera_names + additional_camera_for_voxel

    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=camera_names, 
        camera_height=args.camera_height, 
        camera_width=args.camera_width, 
        reward_shaping=args.shaped,
    )

    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)
    camera_info = None
    if is_robosuite_env:
        camera_info = env.get_camera_info(
            camera_names=camera_names, 
            camera_height=args.camera_height, 
            camera_width=args.camera_width,
        )

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    num_workers = args.num_workers
    
    
    for i in range(0, len(demos), num_workers):
        end = min(i + num_workers, len(demos))
        initial_state_list = []
        states_list = []
        actions_list = []
        for j in range(i, end):
            ep = demos[j]
            # prepare initial state to reload from
            states = f["data/{}/states".format(ep)][()]
            initial_state = dict(states=states[0])
            if is_robosuite_env:
                xml_str = f["data/{}".format(ep)].attrs["model_file"]
                xml_str = replace_all_lights(xml_str)
                xml_str = recolor_robot(xml_str)
                xml_str = recolor_gripper(xml_str)
                initial_state["model"] = xml_str
            actions = f["data/{}/actions".format(ep)][()]

            initial_state_list.append(initial_state)
            states_list.append(states)
            actions_list.append(actions)
            
        with multiprocessing.Pool(num_workers) as pool:
            trajs = pool.map(worker, [[env_meta, args, camera_names, initial_state_list[j], states_list[j], actions_list[j]] for j in range(len(initial_state_list))]) 

        for j, ind in enumerate(range(i, end)):
            ep = demos[ind]
            traj = trajs[j]
            exclude_cameras_from_obs(traj, additional_camera_for_voxel, store_voxel, store_point_cloud)
            # maybe copy reward or done signal from source file
            if args.copy_rewards:
                traj["rewards"] = f["data/{}/rewards".format(ep)][()]
            if args.copy_dones:
                traj["dones"] = f["data/{}/dones".format(ep)][()]

            # store transitions

            # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
            #            consistent as well
            ep_data_grp = data_grp.create_group(ep)
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            for k in traj["obs"]:
                if args.compress:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]), compression="gzip")
                else:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                if not args.exclude_next_obs:
                    if args.compress:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]), compression="gzip")
                    else:
                        ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if is_robosuite_env:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]
            print("ep {}: wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))
        
        del trajs

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="(optional) num_workers for parallel saving",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=128,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=128,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag to exclude next obs in dataset
    parser.add_argument(
        "--exclude-next-obs", 
        action='store_true',
        help="(optional) exclude next obs in dataset",
    )

    # flag to compress observations with gzip option in hdf5
    parser.add_argument(
        "--compress", 
        action='store_true',
        help="(optional) compress observations with gzip option in hdf5",
    )

    # flag to save voxels in hdf5
    parser.add_argument(
        "--store_voxel", 
        action='store_true',
        help="(optional) save voxels in dataset",
    )
    
    parser.add_argument(
        "--store_point_cloud", 
        action='store_true',
        help="(optional) save point clouds in dataset",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
