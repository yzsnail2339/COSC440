#!/bin/bash

# 定义日志和PID文件路径
LOG_FILE="$HOME/.jupyter/jupyter.log"
PID_FILE="$HOME/.jupyter/jupyter.pid"

# 启动 Jupyter Lab 并将 PID 保存到文件中
nohup jupyter lab  > "$LOG_FILE" 2>&1 &

# 保存 PID
echo $! > "$PID_FILE"

echo "Jupyter Lab 启动成功，日志文件: $LOG_FILE, PID文件: $PID_FILE"