CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather_v2_1280.yml
nohup python -u model_run.py $CONFIG > train_1280.log 2>&1 &
