
import os
import subprocess
from nat.tool.search_tools_base import RAGVectorDB
from pydantic import Field
from typing import Annotated, Any, Dict, List, Tuple

rag_db = RAGVectorDB(persist_directory="./my_vectordb")
rag_db.create_vectorstore()

from ....xzs_config import API_KEY
from starlette.datastructures import Headers

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.settings.global_settings import GlobalSettings
from pydantic import BaseModel, Field, SecretStr
from langchain_core.tools import tool
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.prebuilt import create_react_agent

class SearchToolConfig(FunctionBaseConfig, name="search_local"):
    """
    工具检索专家，从本地工具和mcp服务器寻找能够解决当前用户问题的工具并执行将最终结果返回给用户
    """
    pass
class RequireInput(BaseModel):
    req: str = Field(..., description="用户对于需要生成的代码的功能需求")

@tool
def execute_script(
    file_path: Annotated[str, "包含要执行的Python脚本的完整路径，必须是上一步保存的文件路径"],
    parameters_list:Annotated[List[Tuple[str, str]], "传入代码的参数，参数用二元组格式存放在list中，第一个元素为传入参数名称，第二个元素为参数值"
                                            "示例：[('param_1','123')]"],
) -> str:
    """
    挑选并执行工具并向用户返回执行结果
    
    参数:
        file_path: 调用工具文件的完整路径
    
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
    
    # 添加参数到命令行
        for param_name, param_value in parameters_list:  # parameters_list 是二元组列表
            cmd.extend([f"--{param_name}", str(param_value)])
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
        if output:
            return f"脚本执行成功，输出:\n{output}"
        else:
            return "脚本执行成功，无输出内容"
                
    except subprocess.CalledProcessError as e:
        return f"脚本执行出错，返回码 {e.returncode}:\n错误输出: {e.stderr}"
    except Exception as e:
        with open (file_path,'r',encoding='utf-8')as f:
            content=f.read()
        return f"原始代码：{content}   执行过程出错: {str(e)}"

@register_function(config_type=SearchToolConfig)
async def search(_config: SearchToolConfig, _builder: Builder):

    async def _search_and_deal(user_req: str) -> str:
        

        try:
            final_res=''
            scored_results =rag_db.similarity_search(user_req,3)
            for doc in scored_results:
                final_res+=doc.page_content
            print("03",final_res)
            api_key = SecretStr(API_KEY)
            base_llm = ChatTongyi(model="qwen-plus", api_key=api_key)

            tools=[execute_script]
            agent = create_react_agent(base_llm, tools)
            input_message = {"role": "user", "content":f"""你是一位工具调度专家负责为用户选择适合的工具并完成任务：
                             用户请求：{user_req}
                             这些是寻找到的可能与任务相关的工具信息：{final_res}\n\n\n

                             注意：1."传入代码的参数的存储格式，参数用二元组格式存放在list中，第一个元素为传入参数名称，第二个元素为参数解释"
                                            "示例：[('param_1','param_1是XXX')]"
                                2.当不存在符合要求的工具时告知已有工具不能满足需求，请将用户需求告诉工具制作专家
                             请从其中挑选最合适的工具执行"""}
            res=''
            for step in agent.stream({"messages": [input_message]}, stream_mode="values"):
                step["messages"][-1].pretty_print()
                res=step["messages"][-1].content

            return f"生成结果：{res}"
        except Exception as e:
            return f"{e}"


    yield FunctionInfo.from_fn(
        _search_and_deal,
        description="检索本地是否存在已有工具能够满足任务需求，并调用工具解决任务")
