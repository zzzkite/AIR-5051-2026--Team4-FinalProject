# 远程连接
```bash
ssh radxa@10.31.171.185
```

# 编译
```bash
cd ~/user/workspace
colcon build --packages-select llm_bridge --symlink-install
```

# LLM - ROS2 启动流程并测试
## 1. 启动LLM后台服务
确保你已经激活 conda 环境并启动服务：
```bash
conda activate llm
cd ~/ai-engine-direct-helper/samples
python genie/python/GenieAPIService.py --modelname "DeepSeek-R1-Distill-Qwen-7B" --loadmodel --profile
```
等待看到 `Uvicorn running on http://0.0.0.0:8910`，说明模型已加载，API 服务就绪。


## 2. 运行桥接节点
由于节点和 API 服务都在同一台 Airbox 上，IP 可以用 `127.0.0.1`。

另开一个终端，启动桥接节点
```bash
# 进入工作空间并 source 环境
cd ~/user/workspace
source install/setup.bash

# 运行节点（注意：需要退出 conda 环境，或确保 libcurl 可用）
ros2 run llm_bridge bridge_node --ros-args -p airbox_ip:=127.0.0.1
```

## 3. 发送测试指令
另开一个终端，先导入必要的`bash`，再发布话题：
```bash
source /opt/ros/jazzy/setup.bash
source ~/user/workspace/install/setup.bash

ros2 topic pub /llm_command std_msgs/msg/String "{data: 'move to x=0.2, y=0, z=0.3'}" --once
```


```cpp
用户输入 (/llm_command) → 桥接节点 → HTTP POST 请求 → GenieAPI (Airbox) → LLM 推理
                                                                          ↓
机械臂控制 (/arm_control) ← 桥接节点 ← 解析 JSON ← HTTP 响应 ←───────────────┘
```


# v2
## 启动LLM后台服务
在高通NUC上启动LLM后台服务：
```bash
conda activate llm
cd ~/ai-engine-direct-helper/samples
python genie/python/GenieAPIService.py --modelname "DeepSeek-R1-Distill-Qwen-7B" --loadmodel --profile
```
等待看到 `Uvicorn running on http://0.0.0.0:8910`，说明模型已加载，API 服务就绪。


## 启动PC桥接脚本
PC上开一个终端：
```bash
cd ./team_4
source devel/setup.bash
roslaunch llm_pkg llm_bridge.launch
```

## 启动json解读并且yolo检测的脚本
PC上新开一个终端：
```bash
source devel/setup.bash
roslaunch arm_vision_executor vision_executor.py
```

