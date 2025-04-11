import sys

# Iterate over all directories within the root path

from concurrent import futures
import logging
import grpc
from gen import llm_pb2, llm_pb2_grpc
from openai import OpenAI
class LLMServiceServicer(llm_pb2_grpc.LLMServiceServicer):

    def __init__(self):
        # 初始化OpenLLM模型
        self.client = OpenAI(base_url='https://api.deepseek.com', api_key='')
    
    def Generate(self, request, context):
        # 处理普通生成请求
        try:
            result = self.client.chat.completions.create(
                model='deepseek-chat',
                messages=[
                    {
                        "role":"user",
                        "content":request.prompt,
                    }
                ],
                stream=False,
            )
            return llm_pb2.GenerateResponse(reply=result.choices[0].message.content or "", is_final=True)
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return llm_pb2.GenerateResponse()
    
    def StreamGenerate(self, request, context):
        # 处理流式生成请求
        try:
            stream = self.model.generate_stream(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                history=request.history,
                **request.metadata
            )
            
            for output in stream:
                yield llm_pb2.GenerateResponse(
                    reply=output.text,
                    is_final=False
                )
            
            # 发送最终确认
            yield llm_pb2.GenerateResponse(is_final=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    llm_pb2_grpc.add_LLMServiceServicer_to_server(
        LLMServiceServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()