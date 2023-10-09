protobuf_generate_cpp ( PROTO_SRC PROTO_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/glinthawk.proto )
add_library ( glinthawk_proto ${PROTO_HEADER} ${PROTO_SRC} )
