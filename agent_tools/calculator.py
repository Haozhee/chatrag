from langchain.tools import tool

@tool("calculator", return_direct=True)
def calculator(expr: str) -> str:
    """计算简单算术表达式，如 "3 * (7 + 4)"""    
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {e}"
    