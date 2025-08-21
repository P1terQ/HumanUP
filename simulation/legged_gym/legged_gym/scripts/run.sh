robot_name=${1}
task_name="${robot_name}_up"

proj_name="${robot_name}_up"
exptid=${2}

# Run the training script
python train.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --device "${3}" \
                --num_envs 4096 \
                --headless \
                --fix_action_std \
                # --debug
                # --resume \
                # --resumeid XXX
