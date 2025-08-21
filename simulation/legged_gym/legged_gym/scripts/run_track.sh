robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_track"

proj_name="${robot_name}_track"
exptid=${2}
traj_name=${4}

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${3}" \
                --num_envs 4096 \
                --headless \
                --fix_action_std \
                --traj_name "${traj_name}"\
                # --debug \
                # --resume \
                # --resumeid XXX
