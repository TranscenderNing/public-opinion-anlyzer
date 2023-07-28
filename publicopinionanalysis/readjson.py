import json

# 打开并读取 JSON 文件
with open('finedata.json', 'r') as file:
    json_data = file.read()

# 解析 JSON 数据
parsed_data = json.loads(json_data)

# 输出解析后的数据
print(type(parsed_data),len(parsed_data))

