# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: laser_scan.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x10laser_scan.proto\"\x7f\n\x0cLaserScanMsg\x12\x11\n\ttimestamp\x18\x01 \x01(\x01\x12\x10\n\x08robot_id\x18\x02 \x01(\t\x12\x11\n\tangle_min\x18\x03 \x01(\x02\x12\x11\n\tangle_max\x18\x04 \x01(\x02\x12\x14\n\x0cnum_readings\x18\x05 \x01(\x05\x12\x0e\n\x06values\x18\x06 \x03(\x02\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'laser_scan_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_LASERSCANMSG']._serialized_start=20
  _globals['_LASERSCANMSG']._serialized_end=147
# @@protoc_insertion_point(module_scope)
