from protobuf import glinthawk_pb2 as protobuf

Platform = protobuf.Hey.Platform
Stage = protobuf.SetRoute.LayerToAddress.Stage
Kernel = protobuf.Hey.Kernel

Stage_Type = type(Stage.PreAttention)
Platform_Type = type(Platform.CUDA)
Kernel_Type = type(Kernel.Batched)

stages_in_order = [
    Stage.PreAttention,
    Stage.Attention,
    Stage.PostAttention,
    Stage.Classification,
]


def vector_index(layer: int, stage: Stage_Type):
    return len(stages_in_order) * layer + stages_in_order.index(stage)
