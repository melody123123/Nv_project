import re
import subprocess
import os
from typing import Annotated, Any, Dict, List, Tuple
import json
from pathlib import Path
from nat.tool.search_tools_base import RAGVectorDB
from pydantic import BaseModel, Field, SecretStr
from ....xzs_config import API_KEY
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
class AdvancedAdditionToolConfig(FunctionBaseConfig, name="coding"):
    """
    高级加法工具配置，支持多种输入格式和选项
    """
    pass



@tool
def convent_code(
    code: Annotated[str, "从用户输入中提取出的准备保存的python代码内容，包含完整的实现代码"],
    params:Annotated[List[Tuple[str, str]], "传入代码的参数，参数用二元组格式存放在list中，第一个元素为传入参数名称，第二个元素为这个参数的具体说明"
                                            "示例：[('param_1','关于参数作用的描述')]"],
) -> Tuple[str,List[Tuple[str, str]]]:
    """
    修复代码格式
    
    参数:
        code: 从对话中提取的完整Python代码内容
        params: 传入python代码的参数及参数描述，参数数量不固定，标准格式即可，禁止用argsparse格式
    
    返回:
        str: 提取之后的完整python代码
    """
    # 从代码块中提取纯代码（处理可能的Markdown格式）
    code_block_match = re.search(r'```python(.*?)```', code, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1).strip()
    print("02",params)
    
    return code,params

@tool
def create_code(
    requirements: Annotated[str, "详细的用户需求"]

) -> str:
    """
    根据用户需求生成完整的Python可执行代码。
    
    Args:
        requirements: 用户对代码功能的详细需求描述
        
    Returns:
        str: 生成的完整Python代码字符串
    """
    prompt = f"""你是一个python语言方面的编程专家，请根据以下用户需求生成对应脚本：
    {requirements}
    
    请确保：
    1. 代码是完整可运行的独立脚本
    2. 包含适当的注释
    3. 必须有可执行的主程序逻辑，不需要手动调用函数
    4. 如果使用函数，确保在脚本末尾有调用逻辑，例如：
       if __name__ == "__main__":
           main()
    5. 不要包含任何解释性文字，只返回代码
    6. 如需传入参数传入参数在主函数中用argparse传入，并添加注释
    7. 验证并确保程序可用
    """
    api_key = SecretStr(API_KEY)
    base_llm = ChatTongyi(model="qwen-plus", api_key=api_key)
    # 生成代码
    response = base_llm.invoke([HumanMessage(content=prompt)])
    code_content = response.content
    return str(code_content)

@tool
def save_file(
    code: Annotated[str, "从用户输入中提取出的准备保存的python代码内容，包含完整的实现代码，如果存在传入参数则在主函数部分用args传入"],
    
    file_name: Annotated[str, "根据代码内容和用户请求总结最适合的文件名，应该包含.py扩展名，使用蛇形命名法，如：get_system_ip.py"]
) -> str:
    """
    保存Python代码文件到指定位置。根据代码内容自动生成合适的文件名，确保文件名具有描述性和一致性。
    
    参数:
        code: 从对话中提取的完整Python代码内容
        
        file_name: 基于代码功能和用户请求生成的用于保存文件的描述性文件名
        
    返回:
        str: 实际保存的文件名，包含完整路径信息
    
    示例:
        save_file("import os\nprint('hello')", "hello_world.py")
    """
    # 确保文件名有.py扩展名
    if not file_name.endswith('.py'):
        file_name = f"{file_name}.py"
    
    # 清理文件名中的非法字符
    file_name = re.sub(r'[<>:"/\\|?*]', '_', file_name)
    
    # 创建保存目录（如果不存在）
    save_dir = Path("./saved_scripts")
    save_dir.mkdir(exist_ok=True)
    
    # 构建完整文件路径
    file_path = save_dir / file_name
    
    # 保存文件内容
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(code)
        
        # 返回保存成功的确认信息 - 
        
        print(f"文件路径：{file_path.absolute()}")
        return f"文件路径：{file_path.absolute()}"
    
    except Exception as e:
        return f"文件保存失败: {str(e)}"


@tool
def execute_script(
    file_path: Annotated[str, "包含要执行的Python脚本的完整路径，必须是上一步保存的文件路径"],
    parameters_list:Annotated[List[Tuple[str, str,str ]], "传入代码的参数，参数用三元组格式存放在list中，第一个元素为传入参数名称，第二个元素为参数值"
                                            "示例：[('param_1','123','param_1是XXX')]"],
    description: Annotated[str, "对代码功能的详细描述"],
) -> str:
    """
    这是一个脚本功能测试程序，检测程序执行是否成功，直接执行Python脚本文件，就像在命令行中运行python file.py一样，
    
    参数:
        file_path: 上一步保存的Python文件的完整路径
    
    返回:
        str: 脚本执行结果或错误信息
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return f"错误: 文件不存在 - {file_path}"
            
        # 检查是否是Python文件
        if not file_path.endswith('.py'):
            return f"错误: 不是Python文件 - {file_path}"
        cmd = ['python', file_path]
        params=[]
    
    # 添加参数到命令行
        for param_name, param_value,param_description in parameters_list:  # parameters_list 是二元组列表
            cmd.extend([f"--{param_name}", str(param_value)])
            params.append((param_name,param_description))
        # 使用subprocess直接执行脚本，捕获输出和错误
        print(cmd)
        result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
        
        # 返回执行结果
        output = result.stdout.strip()
        rag_db = RAGVectorDB(persist_directory="./my_vectordb")
        rag_db.create_vectorstore()
        rag_db.add_single_document(
        text= f"""
        工具作用：{description}
        程序路径：{file_path}
        传入参数：{params}
""",     metadata={"source": "local", "type": "agent"})
    
        if output:
            return f"脚本执行成功，输出:\n{output}"
        else:
            return "脚本执行成功，无输出内容"
                
    except subprocess.CalledProcessError as e:
        with open (file_path,'r',encoding='utf-8')as f:
            content=f.read()
        
        return f"原始代码：{content} \n 脚本执行出错，返回码 {e.returncode}:\n错误输出: {e.stderr}"
    except Exception as e:
        with open (file_path,'r',encoding='utf-8')as f:
            content=f.read()
        return f"原始代码：{content}   执行过程出错: {str(e)}"


class AdditionInput(BaseModel):
    req: str = Field(..., description="用户对于需要生成的代码的功能需求")
    execute: bool = Field(True, description="是否在生成代码后执行该脚本，默认为True")


from langchain_core.runnables import RunnableConfig


def coding_tool(requirements: str, execute: bool = True):
    """
    使用指定的创意LLM进行代码生成，并直接执行生成的Python脚本
    
    Args:
        requirements: 代码功能需求
        execute: 是否执行生成的代码
    
    Returns:
        str: 生成的内容及执行结果
    """
    try:
        api_key = SecretStr(xzs_config.API_KEY)
        base_llm = ChatTongyi(model="qwen-plus", api_key=api_key)
        memory = MemorySaver() 
        config: RunnableConfig = {"configurable": {"thread_id": "abc123"}}

        tools=[save_file,convent_code,execute_script,create_code]
        agent = create_react_agent(base_llm, tools, checkpointer=memory)
        input_message = {"role": "user", "content":f"""你是一个python语言方面的编程专家，请根据以下用户需求调用工具生成对应脚本并自行验证程序是否可用，
                    若不可用则重新生成：
        {requirements}
        """}
        res=''
        for step in agent.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
            step["messages"][-1].pretty_print()
            res+=step["messages"][-1].content

        return f"生成结果：{res}"
    finally:
        # memory.clear(config)
        pass


@register_function(config_type=AdvancedAdditionToolConfig)
async def advanced_addition(config: AdvancedAdditionToolConfig, _builder: Builder):
    """
    工具生成，直接执行保存的Python脚本文件
    """
    
    async def _advanced_add(inputs: AdditionInput) -> str:
        try:
            # 传递用户需求和执行参数
            return f"生成代码及执行结果：{coding_tool(inputs.req, inputs.execute)}"
                
        except Exception as e:
            error_msg = f"代码生成工具出错: {str(e)}"
            return error_msg
    
    yield FunctionInfo.from_fn(
        _advanced_add,
        description="根据用户需求生成对应工具的python代码，并直接执行该脚本文件"
    )
    