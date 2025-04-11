import grpc
from gen import llm_pb2, llm_pb2_grpc

class LLMClient:
    def __init__(self, host='localhost', port=50051):
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = llm_pb2_grpc.LLMServiceStub(self.channel)
    
    def generate(self, prompt, max_tokens=100, temperature=0.7, history=None, metadata=None):
        request = llm_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            history=history or [],
            metadata=metadata or {}
        )
        response = self.stub.Generate(request)
        return response.reply
    
    def stream_generate(self, prompt, max_tokens=100, temperature=0.7, history=None, metadata=None):
        request = llm_pb2.GenerateRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            history=history or [],
            metadata=metadata or {}
        )
        
        responses = self.stub.StreamGenerate(request)
        for response in responses:
            if response.reply:
                yield response.reply
            if response.is_final:
                break

if __name__ == '__main__':
    client = LLMClient()
    
    # 测试同步调用
    print("Testing synchronous generate:")
    response = client.generate("Hello, introduce yourself")
    print(f"Response: {response}")
    
 