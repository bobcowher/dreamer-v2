source ~/anaconda3/etc/profile.d/conda.sh

conda activate dreamer-gemini

# python ./train.py
python ./actor_critic.py
# python ./rssm.py
# python ./world_model.py
# python ./image_dump.py
# python ./human_demo_w_controller.py
