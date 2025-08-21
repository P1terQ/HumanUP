robot_name=${1}  # Remove the space around the assignment operator
task_name="${robot_name}_up"

proj_name="${robot_name}_up"
exptid=${2}
checkpoint=${3}

python play.py --task "${task_name}" \
                --proj_name "${proj_name}" \
                --exptid "${exptid}" \
                --num_envs 1 \
                --checkpoint "${checkpoint}" \
                --record_video
