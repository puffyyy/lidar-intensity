CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather_0_v2_640.yml
time=`date -I`
python -u model_eval.py $CONFIG > eval_weather_640_0_new.log 2>&1 
CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather_1_v2_640.yml
time=`date -I`
python -u model_eval.py $CONFIG > eval_weather_640_1_new.log 2>&1 
CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather_2_v2_640.yml
time=`date -I`
python -u model_eval.py $CONFIG > eval_weather_640_2_new.log 2>&1 
CONFIG=/home/admin/workspace/yangdonglin/workplace/lidar-intensity/configs/train.reflect-l2.depth.rgb.label.weather_3_v2_640.yml
time=`date -I`
python -u model_eval.py $CONFIG > eval_weather_640_3_new.log 2>&1 