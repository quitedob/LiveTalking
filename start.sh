# 文件路径：/workspace/LiveTalking/start.sh
#!/usr/bin/env bash

# 启动数字人服务
python app.py --model wav2lip --transport webrtc --avatar_id musetalk_avatar1
#python app.py --model musetalk --transport webrtc --avatar_id musetalk_avatar1
