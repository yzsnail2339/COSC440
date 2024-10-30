#!/bin/bash

# 定义PID文件路径
PID_FILE="$HOME/.jupyter/jupyter.pid"

# 检查PID文件是否存在
if [ ! -f "$PID_FILE" ]; then
  echo "PID 文件不存在: $PID_FILE"
  exit 1
fi

# 读取 PID 并停止进程
PID=$(cat "$PID_FILE")

if ps -p $PID > /dev/null; then
  kill $PID
  echo "Jupyter Lab (PID: $PID) 已停止。"
  rm -f "$PID_FILE"
else
  echo "没有找到运行中的 Jupyter Lab 进程 (PID: $PID)"
fi
