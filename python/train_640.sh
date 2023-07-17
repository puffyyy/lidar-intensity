CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.v2_640.yml
time=`date -I`
nohup python -u model_run.py $CONFIG > train_no_weather_640_${time}.log 2>&1 &
