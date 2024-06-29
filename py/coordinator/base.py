from protobuf import glinthawk_pb2 as protobuf

Platform = protobuf.Hey.Platform
Stage = protobuf.SetRoute.LayerToAddress.Stage
Kernel = protobuf.Hey.Kernel

Stage_Type = type(Stage.PreAttention)
Platform_Type = type(Platform.CUDA)
Kernel_Type = type(Kernel.Batched)
