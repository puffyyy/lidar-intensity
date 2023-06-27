CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather.v2_640.yml
nohup python -u model_run.py $CONFIG > train_640.log 2>&1 &
