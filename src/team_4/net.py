import requests
import json

DEVICE_IP = "10.31.171.185"  # 开发板 IP
PORT = "8910"
MODEL_NAME = "Phi-3.5-mini"  # 确保这与启动时的名称匹配

url = f"http://{DEVICE_IP}:{PORT}/v1/chat/completions"


def chat_with_model(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,  # 开启流式
        "temp": 0.8,
        "top_p": 1.0
    }

    headers = {"Content-Type": "application/json"}

    try:
        # 使用 stream=True 发起请求
        response = requests.post(url, json=payload, headers=headers, stream=True)

        if response.status_code != 200:
            print(f"错误码: {response.status_code}, 响应: {response.text}")
            return

        print(f"模型 ({MODEL_NAME}) 回答: ", end="", flush=True)

        # 解析流式 SSE 数据
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith("data: "):
                    data_str = line_text[6:]  # 去掉 "data: " 前缀

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                        # 获取内容增量
                        content = chunk['choices'][0]['delta'].get('content', '')
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
        print("\n\n[任务完成]")

    except Exception as e:
        print(f"\n连接失败: {e}")

if __name__ == "__main__":
    user_input = "请简述机器人研究的三个前沿趋势。"
    chat_with_model(user_input)

