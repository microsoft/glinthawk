# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: glinthawk.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fglinthawk.proto\x12\x12glinthawk.protobuf\"\x7f\n\x10InitializeWorker\x12\x12\n\nmodel_name\x18\x01 \x01(\t\x12\x13\n\x0bstart_layer\x18\x02 \x01(\r\x12\x11\n\tend_layer\x18\x03 \x01(\r\x12\x18\n\x10\x63oncurrency_size\x18\x04 \x01(\r\x12\x15\n\rblobstore_uri\x18\x05 \x01(\t\"$\n\x0eProcessPrompts\x12\x12\n\nprompt_ids\x18\x01 \x03(\t\"%\n\x0fPromptCompleted\x12\x12\n\nprompt_ids\x18\x01 \x03(\t\"\x90\x01\n\x08SetRoute\x12\x45\n\x10layer_to_address\x18\x01 \x03(\x0b\x32+.glinthawk.protobuf.SetRoute.LayerToAddress\x1a=\n\x0eLayerToAddress\x12\x11\n\tlayer_num\x18\x01 \x01(\r\x12\n\n\x02ip\x18\x02 \x01(\t\x12\x0c\n\x04port\x18\x03 \x01(\r\"\xa4\x01\n\x0bWorkerStats\x12\x17\n\x0fstates_received\x18\x01 \x01(\x04\x12\x13\n\x0bstates_sent\x18\x02 \x01(\x04\x12\x18\n\x10states_processed\x18\x03 \x01(\x04\x12\x18\n\x10tokens_processed\x18\x04 \x01(\x04\x12\x18\n\x10tokens_generated\x18\x05 \x01(\x04\x12\x19\n\x11prompts_completed\x18\x06 \x01(\x04\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'glinthawk_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _INITIALIZEWORKER._serialized_start=39
  _INITIALIZEWORKER._serialized_end=166
  _PROCESSPROMPTS._serialized_start=168
  _PROCESSPROMPTS._serialized_end=204
  _PROMPTCOMPLETED._serialized_start=206
  _PROMPTCOMPLETED._serialized_end=243
  _SETROUTE._serialized_start=246
  _SETROUTE._serialized_end=390
  _SETROUTE_LAYERTOADDRESS._serialized_start=329
  _SETROUTE_LAYERTOADDRESS._serialized_end=390
  _WORKERSTATS._serialized_start=393
  _WORKERSTATS._serialized_end=557
# @@protoc_insertion_point(module_scope)
