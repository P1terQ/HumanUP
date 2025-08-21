robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_track"

proj_name="${robot_name}_track"
exptid=${2}
checkpoint=${3}
traj_name=${4}

python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --checkpoint "${checkpoint}" \
                --traj_name "${traj_name}"\
                --record_video
