# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# NO CHECKED-IN PROTOBUF GENCODE
# source: serializable_fairprices.proto
# Protobuf Python Version: 5.29.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    5,
    29,
    0,
    '',
    'serializable_fairprices.proto'
)
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1dserializable_fairprices.proto\x12\x0cserializable\"\xa1\x04\n\tFairPrice\x12\x18\n\x10\x64\x65livery_area_id\x18\x01 \x01(\t\x12\x1d\n\x15\x64\x65livery_start_ms_utc\x18\x02 \x01(\x04\x12\x1b\n\x13\x64\x65livery_end_ms_utc\x18\x03 \x01(\x04\x12\x16\n\x0elegacy_avg_bid\x18\x04 \x01(\x11\x12\x17\n\x0flegacy_best_bid\x18\x05 \x01(\x11\x12\x19\n\x11legacy_fair_price\x18\x06 \x01(\x11\x12\x17\n\x0flegacy_best_ask\x18\x07 \x01(\x11\x12\x16\n\x0elegacy_avg_ask\x18\x08 \x01(\x11\x12\x18\n\x0bmax_5mw_bid\x18\n \x01(\x11H\x00\x88\x01\x01\x12\x15\n\x08\x62\x65st_bid\x18\x0b \x01(\x11H\x01\x88\x01\x01\x12\x17\n\nfair_price\x18\x0c \x01(\x11H\x02\x88\x01\x01\x12\x15\n\x08\x62\x65st_ask\x18\r \x01(\x11H\x03\x88\x01\x01\x12\x18\n\x0bmax_5mw_ask\x18\x0e \x01(\x11H\x04\x88\x01\x01\x12.\n\x06status\x18\t \x01(\x0e\x32\x1e.serializable.FairPrice.Status\x12\x13\n\x06source\x18\x0f \x01(\tH\x05\x88\x01\x01\"-\n\x06Status\x12\x0b\n\x07UNKNOWN\x10\x00\x12\t\n\x05VALID\x10\x01\x12\x0b\n\x07INVALID\x10\x02\x42\x0e\n\x0c_max_5mw_bidB\x0b\n\t_best_bidB\r\n\x0b_fair_priceB\x0b\n\t_best_askB\x0e\n\x0c_max_5mw_askB\t\n\x07_source\"W\n\nFairPrices\x12\x1e\n\x16generation_time_ms_utc\x18\x01 \x01(\x04\x12)\n\x08products\x18\x02 \x03(\x0b\x32\x17.serializable.FairPriceb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'serializable_fairprices_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_FAIRPRICE']._serialized_start=48
  _globals['_FAIRPRICE']._serialized_end=593
  _globals['_FAIRPRICE_STATUS']._serialized_start=464
  _globals['_FAIRPRICE_STATUS']._serialized_end=509
  _globals['_FAIRPRICES']._serialized_start=595
  _globals['_FAIRPRICES']._serialized_end=682
# @@protoc_insertion_point(module_scope)
