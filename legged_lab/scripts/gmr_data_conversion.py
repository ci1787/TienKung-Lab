
# import pickle
# import numpy as np
# import torch
# import argparse
# from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat  
# from scipy.spatial.transform import Rotation 


# def convert_pkl_to_custom(input_pkl, output_txt, fps):
#     dt = 1.0 / fps

#     with open(input_pkl, "rb") as f:
#         motion_data = pickle.load(f)

#     # root_pos = motion_data["root_pos"]
#     # root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
#     # dof_pos = motion_data["dof_pos"]

#     # Ensure all data are numpy arrays (robust to list or ndarray input)
#     root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
#     root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)  # assumed xyzw
#     dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)
#     # Convert rotation from xyzw (saved format) to wxyz (Isaac Lab convention)
#     root_rot = root_rot[:, [3, 0, 1, 2]]  # xyzw → wxyz




#     root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt
#     root_rot_t = torch.tensor(root_rot, dtype=torch.float32)

#     q1_conj = quat_conjugate(root_rot_t[:-1])         
#     dq = quat_mul(q1_conj, root_rot_t[1:])            
#     axis_angle = axis_angle_from_quat(dq)             
#     root_ang_vel = axis_angle / dt

#     dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

#     euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
#     euler_angles = np.unwrap(euler_angles, axis=0)

#     data_output = np.concatenate(
#         (root_pos[:-1], euler_angles, dof_pos[:-1],  
#          root_lin_vel, root_ang_vel, dof_vel),
#         axis=1
#     )

#     np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
#     with open(output_txt, 'r') as f:
#         frames_data = f.readlines()

#     frames_data_len = len(frames_data)
#     with open(output_txt, 'w') as f:
#         f.write('{\n')
#         f.write('"LoopMode": "Wrap",\n')
#         f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
#         f.write('"EnableCycleOffsetPosition": true,\n')
#         f.write('"EnableCycleOffsetRotation": true,\n')
#         f.write('"MotionWeight": 0.5,\n\n')
#         f.write('"Frames":\n[\n')

#         for i, line in enumerate(frames_data):
#             line_start_str = '  ['
#             if i == frames_data_len - 1:
#                 f.write(line_start_str + line.rstrip() + ']\n')
#             else:
#                 f.write(line_start_str + line.rstrip() + '],\n')

#         f.write(']\n}')
#     print(f"✅ Successfully converted {input_pkl} to {output_txt}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_pkl", type=str, required=True)
#     parser.add_argument("--output_txt", type=str, required=True)
#     parser.add_argument("--fps", type=float, default=30.0)
#     args = parser.parse_args()

#     convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)
import pickle
import numpy as np
import torch
import argparse
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat  
from scipy.spatial.transform import Rotation 


def convert_pkl_to_custom(input_pkl, output_txt, fps):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    # === 加载原始数据 ===
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot_xyzw = np.asarray(motion_data["root_rot"], dtype=np.float32)  # assumed xyzw
    dof_pos = np.asarray(motion_data["dof_pos"], dtype=np.float32)

    # === 定义你想要的输出顺序（语义顺序）===
    desired_output_order = [
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint"
    ]

    # .pkl 中的原始顺序（臂先，腿后）
    pkl_dof_order = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint"
    ]

    # 构建从 pkl_index → desired_output_index 的映射
    remap_to_desired = []
    for target_name in desired_output_order:
        # 在 pkl_dof_order 中找这个关节的位置
        src_idx = pkl_dof_order.index(target_name)
        remap_to_desired.append(src_idx)

    # 重排 dof_pos 到 desired_output_order
    dof_pos = dof_pos[:, remap_to_desired]

    # === 四元数转换：xyzw → wxyz (Isaac Lab convention) ===
    root_rot = root_rot_xyzw[:, [3, 0, 1, 2]]  # now wxyz

    # === 计算速度 ===
    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt

    root_rot_t = torch.tensor(root_rot, dtype=torch.float32)
    q1_conj = quat_conjugate(root_rot_t[:-1])
    dq = quat_mul(q1_conj, root_rot_t[1:])
    axis_angle = axis_angle_from_quat(dq)
    root_ang_vel = (axis_angle / dt).numpy()

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    # === 欧拉角（用于输出）===
    # scipy expects [x,y,z,w], so reorder wxyz → xyzw
    root_rot_scipy = root_rot[:-1, [1, 2, 3, 0]]
    euler_angles = Rotation.from_quat(root_rot_scipy).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    # === 拼接最终输出数据 ===
    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1], root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    # === 保存为 Isaac Lab MotionClip 格式 ===
    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {dt:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            prefix = '  ['
            suffix = ']' if i == len(frames_data) - 1 else '],'
            f.write(prefix + line.rstrip() + suffix + '\n')

        f.write(']\n}')

    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    convert_pkl_to_custom(args.input_pkl, args.output_txt, args.fps)