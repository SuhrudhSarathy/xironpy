# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: pose.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\npose.proto\"\x90\x01\n\x07PoseMsg\x12\x11\n\ttimestamp\x18\x01 \x01(\x01\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12&\n\x08position\x18\x03 \x01(\x0b\x32\x14.PoseMsg.PositionMsg\x12\x13\n\x0borientation\x18\x04 \x01(\x02\x1a#\n\x0bPositionMsg\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'pose_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_POSEMSG']._serialized_start=15
  _globals['_POSEMSG']._serialized_end=159
  _globals['_POSEMSG_POSITIONMSG']._serialized_start=124
  _globals['_POSEMSG_POSITIONMSG']._serialized_end=159
# @@protoc_insertion_point(module_scope)