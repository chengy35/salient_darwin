-include .build_debug/process.d

.build_debug/process.l: \
	.build_debug/process.d \

debug/process: .build_debug/process.l \
	.build_debug/process.o \

