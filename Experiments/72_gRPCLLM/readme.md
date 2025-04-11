#### Install packages
grpcio==1.64.1
grpcio-tools==1.64.1
openai==1.70.0

#### 1 generate python code
python3 -m grpc_tools.protoc --proto_path=. --python_out=./gen --pyi_out=./gen --grpc_python_out=./gen llm.proto


##### start server
python3 llm_server.py

##### start client
python3 llm_client.py


