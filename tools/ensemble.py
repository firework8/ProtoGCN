from mmcv import load, dump
from protogcn.smp import *
import datetime

joint_path = '../work_dirs/ntu60_xsub/j/best_pred.pkl'
bone_path = '../work_dirs/ntu60_xsub/b/best_pred.pkl'
kbone_path = '../work_dirs/ntu60_xsub/k/best_pred.pkl'
joint_motion_path = '../work_dirs/ntu60_xsub/jm/best_pred.pkl'
bone_motion_path = '../work_dirs/ntu60_xsub/bm/best_pred.pkl'
kbone_motion_path = '../work_dirs/ntu60_xsub/km/best_pred.pkl'

joint = load(joint_path)
bone = load(bone_path)
kbone = load(kbone_path)
joint_motion = load(joint_motion_path)
bone_motion = load(bone_motion_path)
kbone_motion = load(kbone_motion_path)

label = load_label('/data/nturgbd/ntu60_3danno.pkl', 'xsub_val')
# label = load_label('/data/nturgbd/ntu60_3danno.pkl', 'xview_val')
# label = load_label('/data/nturgbd/ntu120_3danno.pkl', 'xsub_val')
# label = load_label('/data/nturgbd/ntu120_3danno.pkl', 'xset_val')
# label = load_label('/data/k400/k400_hrnet.pkl', 'val')
# label = load_label('/data/finegym/gym_hrnet.pkl', 'val')

print('J+B')
fused = comb([joint, bone], [1, 1])
print('Top-1', top1(fused, label))

print('4M')
fused = comb([joint, bone, joint_motion, bone_motion], [2, 2, 1, 1])
print('Top-1', top1(fused, label))

print('6M')
fused = comb([joint, bone, kbone, joint_motion, bone_motion, kbone_motion], [2, 2, 2, 1, 1, 1])
print('Top-1', top1(fused, label))