"""健壮的 JSON 解析 ——json_repair + 正则回退"""

import re
import json_repair

def robust_json_parse(text: str):
    """多层容错解析 LLM 返回的 JSON 字符串

    策略 A: json_repair 直接解析（容忍多余逗号、缺引号等）
    策略 B: 提取 markdown 代码块后再用 json_repair
    策略 C: 正则匹配 JSON 对象或数组
    策略 D: 全都失败 →返回 None

    Args:
        text: LLM 返回的原始文本（可能带 markdown 格式）

    Returns:
          parsed dict/list，失败返回 None
    """

    if not text:
        return None
    
    # 策略 A: 直接 json_repair
    try:
        result = json_repair.load(text)
        if result is not None:
            return result
    except Exception:
        pass

    # 策略 B: 尝试提取 ```json ... ``` 代码块
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_block:
        try:
            result = json_repair.loads(code_block.group(1))
            if result is not None:
                return result
        except Exception:
            pass 
    
    # 策略 C: 正则匹配最外层 JSON 对象
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            return json_repair.loads(obj_match.group(0))
        except Exception:
            pass
    
    # 策略 D: 正则匹配最外层 JSON 数组
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if arr_match:
        try:
            return json_repair.loads(arr_match.group(0))
        except Exception:
            pass

    return None