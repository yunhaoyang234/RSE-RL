??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H? *
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	H? *
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:? *
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*'
_output_shapes
:H?*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:H*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:0H*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:0*
dtype0
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0**
shared_nameconv2d_transpose_2/kernel
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_4/kernel
?
-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
V
0
1
2
3
4
5
$6
%7
*8
+9
010
111
V
0
1
2
3
4
5
$6
%7
*8
+9
010
111
 
?
		variables
6layer_metrics

trainable_variables
7non_trainable_variables
regularization_losses
8layer_regularization_losses
9metrics

:layers
 
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
;layer_metrics
trainable_variables
<non_trainable_variables
regularization_losses
=layer_regularization_losses
>metrics

?layers
 
 
 
?
	variables
@layer_metrics
trainable_variables
Anon_trainable_variables
regularization_losses
Blayer_regularization_losses
Cmetrics

Dlayers
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
	variables
Elayer_metrics
trainable_variables
Fnon_trainable_variables
regularization_losses
Glayer_regularization_losses
Hmetrics

Ilayers
ec
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
 	variables
Jlayer_metrics
!trainable_variables
Knon_trainable_variables
"regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
ec
VARIABLE_VALUEconv2d_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
&	variables
Olayer_metrics
'trainable_variables
Pnon_trainable_variables
(regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
ec
VARIABLE_VALUEconv2d_transpose_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
?
,	variables
Tlayer_metrics
-trainable_variables
Unon_trainable_variables
.regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
ec
VARIABLE_VALUEconv2d_transpose_4/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_4/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
?
2	variables
Ylayer_metrics
3trainable_variables
Znon_trainable_variables
4regularization_losses
[layer_regularization_losses
\metrics

]layers
 
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????H*
dtype0*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2dense_2/kerneldense_2/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_85932
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_86735
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_2/kerneldense_2/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose_4/kernelconv2d_transpose_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_86781??
?
C
'__inference_reshape_layer_call_fn_86481

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_851882
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?A
?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_85125

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_85505

inputs 
dense_2_85383:	H? 
dense_2_85385:	? 1
conv2d_transpose_85389:H?$
conv2d_transpose_85391:H2
conv2d_transpose_1_85394:0H&
conv2d_transpose_1_85396:02
conv2d_transpose_2_85399: 0&
conv2d_transpose_2_85401: 2
conv2d_transpose_3_85404: &
conv2d_transpose_3_85406:2
conv2d_transpose_4_85409:&
conv2d_transpose_4_85411:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_85383dense_2_85385*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_851682!
dense_2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_851882
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_85389conv2d_transpose_85391*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_848852*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_85394conv2d_transpose_1_85396*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_849452,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_85399conv2d_transpose_2_85401*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_850052,
*conv2d_transpose_2/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_85404conv2d_transpose_3_85406*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_850652,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_85409conv2d_transpose_4_85411*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_851252,
*conv2d_transpose_4/StatefulPartitionedCall?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_2_85383*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_85383*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_85389*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_85389*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_85394*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_85394*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_85399*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_85399*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_85404*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_85404*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_85409*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_85409*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?"
?
B__inference_dense_2_layer_call_and_return_conditional_losses_86453

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
Relu?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_86412

inputs
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_855052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?"
?
B__inference_dense_2_layer_call_and_return_conditional_losses_85168

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
Relu?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_3_layer_call_fn_85075

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_850652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?A
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_84945

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :02	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????0*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????02	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????02
Relu?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_85306

inputs 
dense_2_85169:	H? 
dense_2_85171:	? 1
conv2d_transpose_85190:H?$
conv2d_transpose_85192:H2
conv2d_transpose_1_85195:0H&
conv2d_transpose_1_85197:02
conv2d_transpose_2_85200: 0&
conv2d_transpose_2_85202: 2
conv2d_transpose_3_85205: &
conv2d_transpose_3_85207:2
conv2d_transpose_4_85210:&
conv2d_transpose_4_85212:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_85169dense_2_85171*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_851682!
dense_2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_851882
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_85190conv2d_transpose_85192*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_848852*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_85195conv2d_transpose_1_85197*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_849452,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_85200conv2d_transpose_2_85202*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_850052,
*conv2d_transpose_2/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_85205conv2d_transpose_3_85207*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_850652,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_85210conv2d_transpose_4_85212*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_851252,
*conv2d_transpose_4/StatefulPartitionedCall?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_2_85169*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_85169*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_85190*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_85190*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_85195*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_85195*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_85200*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_85200*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_85205*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_85205*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_85210*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_85210*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_86143

inputs9
&dense_2_matmul_readvariableop_resource:	H? 6
'dense_2_biasadd_readvariableop_resource:	? T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:H?>
0conv2d_transpose_biasadd_readvariableop_resource:HU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_1_biasadd_readvariableop_resource:0U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_2/Relu?
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/BiasAdd?
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/Relu?
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slicez
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/1z
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/BiasAdd?
conv2d_transpose_4/SigmoidSigmoid#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/Sigmoid?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?

IdentityIdentityconv2d_transpose_4/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_2_layer_call_fn_85015

inputs!
unknown: 0
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_850052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_85188

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
0__inference_conv2d_transpose_layer_call_fn_84895

inputs"
unknown:H?
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_848852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_86576I
6dense_2_kernel_regularizer_abs_readvariableop_resource:	H? 
identity??-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
IdentityIdentity$dense_2/kernel/Regularizer/add_1:z:0.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_2_86616[
Aconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource:0H
identity??8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_1/kernel/Regularizer/add_1:z:09^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp
??
?
@__inference_model_layer_call_and_return_conditional_losses_86354

inputs9
&dense_2_matmul_readvariableop_resource:	H? 6
'dense_2_biasadd_readvariableop_resource:	? T
9conv2d_transpose_conv2d_transpose_readvariableop_resource:H?>
0conv2d_transpose_biasadd_readvariableop_resource:HU
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_1_biasadd_readvariableop_resource:0U
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_2_biasadd_readvariableop_resource: U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_3_biasadd_readvariableop_resource:U
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_4/BiasAdd/ReadVariableOp?2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_2/BiasAdd?
conv2d_transpose_2/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_2/Relu?
conv2d_transpose_3/ShapeShape%conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/BiasAdd?
conv2d_transpose_3/ReluRelu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_3/Relu?
conv2d_transpose_4/ShapeShape%conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_4/Shape?
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_4/strided_slice/stack?
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_1?
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_4/strided_slice/stack_2?
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_4/strided_slicez
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/1z
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/2z
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_4/stack/3?
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_4/stack?
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_4/strided_slice_1/stack?
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_1?
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_4/strided_slice_1/stack_2?
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_4/strided_slice_1?
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_4/conv2d_transpose?
)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_4/BiasAdd/ReadVariableOp?
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/BiasAdd?
conv2d_transpose_4/SigmoidSigmoid#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_4/Sigmoid?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?

IdentityIdentityconv2d_transpose_4/Sigmoid:y:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_4_layer_call_fn_85135

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_851252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_86383

inputs
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_853062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_86656[
Aconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource: 
identity??8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_3/kernel/Regularizer/add_1:z:09^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp
?
^
B__inference_reshape_layer_call_and_return_conditional_losses_86476

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:?????????? :P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
??
?
@__inference_model_layer_call_and_return_conditional_losses_85686
input_2 
dense_2_85564:	H? 
dense_2_85566:	? 1
conv2d_transpose_85570:H?$
conv2d_transpose_85572:H2
conv2d_transpose_1_85575:0H&
conv2d_transpose_1_85577:02
conv2d_transpose_2_85580: 0&
conv2d_transpose_2_85582: 2
conv2d_transpose_3_85585: &
conv2d_transpose_3_85587:2
conv2d_transpose_4_85590:&
conv2d_transpose_4_85592:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_85564dense_2_85566*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_851682!
dense_2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_851882
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_85570conv2d_transpose_85572*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_848852*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_85575conv2d_transpose_1_85577*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_849452,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_85580conv2d_transpose_2_85582*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_850052,
*conv2d_transpose_2/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_85585conv2d_transpose_3_85587*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_850652,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_85590conv2d_transpose_4_85592*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_851252,
*conv2d_transpose_4/StatefulPartitionedCall?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_2_85564*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_85564*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_85570*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_85570*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_85575*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_85575*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_85580*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_85580*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_85585*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_85585*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_85590*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_85590*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?&
?
__inference__traced_save_86735
file_prefix-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	H? :? :H?:H:0H:0: 0: : :::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	H? :!

_output_shapes	
:? :-)
'
_output_shapes
:H?: 

_output_shapes
:H:,(
&
_output_shapes
:0H: 

_output_shapes
:0:,(
&
_output_shapes
: 0: 

_output_shapes
: :,	(
&
_output_shapes
: : 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?A
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84885

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????H*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????H2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????H2
Relu?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_86676[
Aconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource:
identity??8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_4/kernel/Regularizer/add_1:z:09^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp
?
?
'__inference_dense_2_layer_call_fn_86462

inputs
unknown:	H? 
	unknown_0:	? 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_851682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_86636[
Aconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource: 0
identity??8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_2/kernel/Regularizer/add_1:z:09^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp
?A
?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_85065

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_85932
input_2
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_848352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
??
?
@__inference_model_layer_call_and_return_conditional_losses_85811
input_2 
dense_2_85689:	H? 
dense_2_85691:	? 1
conv2d_transpose_85695:H?$
conv2d_transpose_85697:H2
conv2d_transpose_1_85700:0H&
conv2d_transpose_1_85702:02
conv2d_transpose_2_85705: 0&
conv2d_transpose_2_85707: 2
conv2d_transpose_3_85710: &
conv2d_transpose_3_85712:2
conv2d_transpose_4_85715:&
conv2d_transpose_4_85717:
identity??(conv2d_transpose/StatefulPartitionedCall?6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_1/StatefulPartitionedCall?8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_2/StatefulPartitionedCall?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_3/StatefulPartitionedCall?8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_4/StatefulPartitionedCall?8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?dense_2/StatefulPartitionedCall?-dense_2/kernel/Regularizer/Abs/ReadVariableOp?0dense_2/kernel/Regularizer/Square/ReadVariableOp?
dense_2/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_2_85689dense_2_85691*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_851682!
dense_2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_851882
reshape/PartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_transpose_85695conv2d_transpose_85697*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_848852*
(conv2d_transpose/StatefulPartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_85700conv2d_transpose_1_85702*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_849452,
*conv2d_transpose_1/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_transpose_2_85705conv2d_transpose_2_85707*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_850052,
*conv2d_transpose_2/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_85710conv2d_transpose_3_85712*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_850652,
*conv2d_transpose_3/StatefulPartitionedCall?
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_3/StatefulPartitionedCall:output:0conv2d_transpose_4_85715conv2d_transpose_4_85717*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_851252,
*conv2d_transpose_4/StatefulPartitionedCall?
 dense_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_2/kernel/Regularizer/Const?
-dense_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_2_85689*
_output_shapes
:	H? *
dtype02/
-dense_2/kernel/Regularizer/Abs/ReadVariableOp?
dense_2/kernel/Regularizer/AbsAbs5dense_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_2/kernel/Regularizer/Abs?
"dense_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_1?
dense_2/kernel/Regularizer/SumSum"dense_2/kernel/Regularizer/Abs:y:0+dense_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/Sum?
 dense_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_2/kernel/Regularizer/mul/x?
dense_2/kernel/Regularizer/mulMul)dense_2/kernel/Regularizer/mul/x:output:0'dense_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/mul?
dense_2/kernel/Regularizer/addAddV2)dense_2/kernel/Regularizer/Const:output:0"dense_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_2/kernel/Regularizer/add?
0dense_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_2_85689*
_output_shapes
:	H? *
dtype022
0dense_2/kernel/Regularizer/Square/ReadVariableOp?
!dense_2/kernel/Regularizer/SquareSquare8dense_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_2/kernel/Regularizer/Square?
"dense_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_2/kernel/Regularizer/Const_2?
 dense_2/kernel/Regularizer/Sum_1Sum%dense_2/kernel/Regularizer/Square:y:0+dense_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/Sum_1?
"dense_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_2/kernel/Regularizer/mul_1/x?
 dense_2/kernel/Regularizer/mul_1Mul+dense_2/kernel/Regularizer/mul_1/x:output:0)dense_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/mul_1?
 dense_2/kernel/Regularizer/add_1AddV2"dense_2/kernel/Regularizer/add:z:0$dense_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_2/kernel/Regularizer/add_1?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_85695*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_85695*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
+conv2d_transpose_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_1/kernel/Regularizer/Const?
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_1_85700*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_1/kernel/Regularizer/AbsAbs@conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_1/kernel/Regularizer/Abs?
-conv2d_transpose_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_1?
)conv2d_transpose_1/kernel/Regularizer/SumSum-conv2d_transpose_1/kernel/Regularizer/Abs:y:06conv2d_transpose_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/Sum?
+conv2d_transpose_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_1/kernel/Regularizer/mul/x?
)conv2d_transpose_1/kernel/Regularizer/mulMul4conv2d_transpose_1/kernel/Regularizer/mul/x:output:02conv2d_transpose_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/mul?
)conv2d_transpose_1/kernel/Regularizer/addAddV24conv2d_transpose_1/kernel/Regularizer/Const:output:0-conv2d_transpose_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_1/kernel/Regularizer/add?
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_1_85700*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_1/kernel/Regularizer/SquareSquareCconv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_1/kernel/Regularizer/Square?
-conv2d_transpose_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_1/kernel/Regularizer/Const_2?
+conv2d_transpose_1/kernel/Regularizer/Sum_1Sum0conv2d_transpose_1/kernel/Regularizer/Square:y:06conv2d_transpose_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/Sum_1?
-conv2d_transpose_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_1/kernel/Regularizer/mul_1/x?
+conv2d_transpose_1/kernel/Regularizer/mul_1Mul6conv2d_transpose_1/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/mul_1?
+conv2d_transpose_1/kernel/Regularizer/add_1AddV2-conv2d_transpose_1/kernel/Regularizer/add:z:0/conv2d_transpose_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_1/kernel/Regularizer/add_1?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_2_85705*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_2_85705*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
+conv2d_transpose_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_3/kernel/Regularizer/Const?
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_3_85710*&
_output_shapes
: *
dtype02:
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_3/kernel/Regularizer/AbsAbs@conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Abs?
-conv2d_transpose_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_1?
)conv2d_transpose_3/kernel/Regularizer/SumSum-conv2d_transpose_3/kernel/Regularizer/Abs:y:06conv2d_transpose_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/Sum?
+conv2d_transpose_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_3/kernel/Regularizer/mul/x?
)conv2d_transpose_3/kernel/Regularizer/mulMul4conv2d_transpose_3/kernel/Regularizer/mul/x:output:02conv2d_transpose_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/mul?
)conv2d_transpose_3/kernel/Regularizer/addAddV24conv2d_transpose_3/kernel/Regularizer/Const:output:0-conv2d_transpose_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_3/kernel/Regularizer/add?
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_3_85710*&
_output_shapes
: *
dtype02=
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_3/kernel/Regularizer/SquareSquareCconv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_3/kernel/Regularizer/Square?
-conv2d_transpose_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_3/kernel/Regularizer/Const_2?
+conv2d_transpose_3/kernel/Regularizer/Sum_1Sum0conv2d_transpose_3/kernel/Regularizer/Square:y:06conv2d_transpose_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/Sum_1?
-conv2d_transpose_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_3/kernel/Regularizer/mul_1/x?
+conv2d_transpose_3/kernel/Regularizer/mul_1Mul6conv2d_transpose_3/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/mul_1?
+conv2d_transpose_3/kernel/Regularizer/add_1AddV2-conv2d_transpose_3/kernel/Regularizer/add:z:0/conv2d_transpose_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_3/kernel/Regularizer/add_1?
+conv2d_transpose_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_4/kernel/Regularizer/Const?
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_4_85715*&
_output_shapes
:*
dtype02:
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_4/kernel/Regularizer/AbsAbs@conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_4/kernel/Regularizer/Abs?
-conv2d_transpose_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_1?
)conv2d_transpose_4/kernel/Regularizer/SumSum-conv2d_transpose_4/kernel/Regularizer/Abs:y:06conv2d_transpose_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/Sum?
+conv2d_transpose_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_4/kernel/Regularizer/mul/x?
)conv2d_transpose_4/kernel/Regularizer/mulMul4conv2d_transpose_4/kernel/Regularizer/mul/x:output:02conv2d_transpose_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/mul?
)conv2d_transpose_4/kernel/Regularizer/addAddV24conv2d_transpose_4/kernel/Regularizer/Const:output:0-conv2d_transpose_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_4/kernel/Regularizer/add?
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_4_85715*&
_output_shapes
:*
dtype02=
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_4/kernel/Regularizer/SquareSquareCconv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_4/kernel/Regularizer/Square?
-conv2d_transpose_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_4/kernel/Regularizer/Const_2?
+conv2d_transpose_4/kernel/Regularizer/Sum_1Sum0conv2d_transpose_4/kernel/Regularizer/Square:y:06conv2d_transpose_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/Sum_1?
-conv2d_transpose_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_4/kernel/Regularizer/mul_1/x?
+conv2d_transpose_4/kernel/Regularizer/mul_1Mul6conv2d_transpose_4/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/mul_1?
+conv2d_transpose_4/kernel/Regularizer/add_1AddV2-conv2d_transpose_4/kernel/Regularizer/add:z:0/conv2d_transpose_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_4/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_4/StatefulPartitionedCall:output:0)^conv2d_transpose/StatefulPartitionedCall7^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_1/StatefulPartitionedCall9^conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_2/StatefulPartitionedCall9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_3/StatefulPartitionedCall9^conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_4/StatefulPartitionedCall9^conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp ^dense_2/StatefulPartitionedCall.^dense_2/kernel/Regularizer/Abs/ReadVariableOp1^dense_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2t
8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_1/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_1/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2t
8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_3/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_3/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2t
8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_4/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_4/kernel/Regularizer/Square/ReadVariableOp2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2^
-dense_2/kernel/Regularizer/Abs/ReadVariableOp-dense_2/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_2/kernel/Regularizer/Square/ReadVariableOp0dense_2/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?A
?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_85005

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
+conv2d_transpose_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_2/kernel/Regularizer/Const?
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_2/kernel/Regularizer/AbsAbs@conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_2/kernel/Regularizer/Abs?
-conv2d_transpose_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_1?
)conv2d_transpose_2/kernel/Regularizer/SumSum-conv2d_transpose_2/kernel/Regularizer/Abs:y:06conv2d_transpose_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/Sum?
+conv2d_transpose_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_2/kernel/Regularizer/mul/x?
)conv2d_transpose_2/kernel/Regularizer/mulMul4conv2d_transpose_2/kernel/Regularizer/mul/x:output:02conv2d_transpose_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/mul?
)conv2d_transpose_2/kernel/Regularizer/addAddV24conv2d_transpose_2/kernel/Regularizer/Const:output:0-conv2d_transpose_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_2/kernel/Regularizer/add?
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_2/kernel/Regularizer/SquareSquareCconv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_2/kernel/Regularizer/Square?
-conv2d_transpose_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_2/kernel/Regularizer/Const_2?
+conv2d_transpose_2/kernel/Regularizer/Sum_1Sum0conv2d_transpose_2/kernel/Regularizer/Square:y:06conv2d_transpose_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/Sum_1?
-conv2d_transpose_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_2/kernel/Regularizer/mul_1/x?
+conv2d_transpose_2/kernel/Regularizer/mul_1Mul6conv2d_transpose_2/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/mul_1?
+conv2d_transpose_2/kernel/Regularizer/add_1AddV2-conv2d_transpose_2/kernel/Regularizer/add:z:0/conv2d_transpose_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_2/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_2/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_2/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
%__inference_model_layer_call_fn_85561
input_2
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_855052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?
?
__inference_loss_fn_1_86596Z
?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource:H?
identity??6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
)conv2d_transpose/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)conv2d_transpose/kernel/Regularizer/Const?
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype028
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp?
'conv2d_transpose/kernel/Regularizer/AbsAbs>conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2)
'conv2d_transpose/kernel/Regularizer/Abs?
+conv2d_transpose/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_1?
'conv2d_transpose/kernel/Regularizer/SumSum+conv2d_transpose/kernel/Regularizer/Abs:y:04conv2d_transpose/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/Sum?
)conv2d_transpose/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)conv2d_transpose/kernel/Regularizer/mul/x?
'conv2d_transpose/kernel/Regularizer/mulMul2conv2d_transpose/kernel/Regularizer/mul/x:output:00conv2d_transpose/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/mul?
'conv2d_transpose/kernel/Regularizer/addAddV22conv2d_transpose/kernel/Regularizer/Const:output:0+conv2d_transpose/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2)
'conv2d_transpose/kernel/Regularizer/add?
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOpReadVariableOp?conv2d_transpose_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype02;
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp?
*conv2d_transpose/kernel/Regularizer/SquareSquareAconv2d_transpose/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2,
*conv2d_transpose/kernel/Regularizer/Square?
+conv2d_transpose/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2-
+conv2d_transpose/kernel/Regularizer/Const_2?
)conv2d_transpose/kernel/Regularizer/Sum_1Sum.conv2d_transpose/kernel/Regularizer/Square:y:04conv2d_transpose/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/Sum_1?
+conv2d_transpose/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose/kernel/Regularizer/mul_1/x?
)conv2d_transpose/kernel/Regularizer/mul_1Mul4conv2d_transpose/kernel/Regularizer/mul_1/x:output:02conv2d_transpose/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/mul_1?
)conv2d_transpose/kernel/Regularizer/add_1AddV2+conv2d_transpose/kernel/Regularizer/add:z:0-conv2d_transpose/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose/kernel/Regularizer/add_1?
IdentityIdentity-conv2d_transpose/kernel/Regularizer/add_1:z:07^conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp:^conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2p
6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp6conv2d_transpose/kernel/Regularizer/Abs/ReadVariableOp2v
9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp9conv2d_transpose/kernel/Regularizer/Square/ReadVariableOp
?
?
%__inference_model_layer_call_fn_85333
input_2
unknown:	H? 
	unknown_0:	? $
	unknown_1:H?
	unknown_2:H#
	unknown_3:0H
	unknown_4:0#
	unknown_5: 0
	unknown_6: #
	unknown_7: 
	unknown_8:#
	unknown_9:

unknown_10:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*.
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *I
fDRB
@__inference_model_layer_call_and_return_conditional_losses_853062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2
?8
?
!__inference__traced_restore_86781
file_prefix2
assignvariableop_dense_2_kernel:	H? .
assignvariableop_1_dense_2_bias:	? E
*assignvariableop_2_conv2d_transpose_kernel:H?6
(assignvariableop_3_conv2d_transpose_bias:HF
,assignvariableop_4_conv2d_transpose_1_kernel:0H8
*assignvariableop_5_conv2d_transpose_1_bias:0F
,assignvariableop_6_conv2d_transpose_2_kernel: 08
*assignvariableop_7_conv2d_transpose_2_bias: F
,assignvariableop_8_conv2d_transpose_3_kernel: 8
*assignvariableop_9_conv2d_transpose_3_bias:G
-assignvariableop_10_conv2d_transpose_4_kernel:9
+assignvariableop_11_conv2d_transpose_4_bias:
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp*assignvariableop_2_conv2d_transpose_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp(assignvariableop_3_conv2d_transpose_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2d_transpose_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_transpose_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
2__inference_conv2d_transpose_1_layer_call_fn_84955

inputs!
unknown:0H
	unknown_0:0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_849452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????02

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????H: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_84835
input_2?
,model_dense_2_matmul_readvariableop_resource:	H? <
-model_dense_2_biasadd_readvariableop_resource:	? Z
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:H?D
6model_conv2d_transpose_biasadd_readvariableop_resource:H[
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:0HF
8model_conv2d_transpose_1_biasadd_readvariableop_resource:0[
Amodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource: 0F
8model_conv2d_transpose_2_biasadd_readvariableop_resource: [
Amodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource: F
8model_conv2d_transpose_3_biasadd_readvariableop_resource:[
Amodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:F
8model_conv2d_transpose_4_biasadd_readvariableop_resource:
identity??-model/conv2d_transpose/BiasAdd/ReadVariableOp?6model/conv2d_transpose/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_1/BiasAdd/ReadVariableOp?8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_2/BiasAdd/ReadVariableOp?8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_3/BiasAdd/ReadVariableOp?8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_4/BiasAdd/ReadVariableOp?8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?$model/dense_2/BiasAdd/ReadVariableOp?#model/dense_2/MatMul/ReadVariableOp?
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02%
#model/dense_2/MatMul/ReadVariableOp?
model/dense_2/MatMulMatMulinput_2+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model/dense_2/MatMul?
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp?
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model/dense_2/BiasAdd?
model/dense_2/ReluRelumodel/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
model/dense_2/Reluz
model/reshape/ShapeShape model/dense_2/Relu:activations:0*
T0*
_output_shapes
:2
model/reshape/Shape?
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stack?
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1?
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_slice?
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/1?
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/2?
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
model/reshape/Reshape/shape/3?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shape?
model/reshape/ReshapeReshape model/dense_2/Relu:activations:0$model/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
model/reshape/Reshape?
model/conv2d_transpose/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
:2
model/conv2d_transpose/Shape?
*model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv2d_transpose/strided_slice/stack?
,model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv2d_transpose/strided_slice/stack_1?
,model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv2d_transpose/strided_slice/stack_2?
$model/conv2d_transpose/strided_sliceStridedSlice%model/conv2d_transpose/Shape:output:03model/conv2d_transpose/strided_slice/stack:output:05model/conv2d_transpose/strided_slice/stack_1:output:05model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv2d_transpose/strided_slice?
model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv2d_transpose/stack/1?
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv2d_transpose/stack/2?
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2 
model/conv2d_transpose/stack/3?
model/conv2d_transpose/stackPack-model/conv2d_transpose/strided_slice:output:0'model/conv2d_transpose/stack/1:output:0'model/conv2d_transpose/stack/2:output:0'model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv2d_transpose/stack?
,model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose/strided_slice_1/stack?
.model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose/strided_slice_1/stack_1?
.model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose/strided_slice_1/stack_2?
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose/strided_slice_1?
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype028
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp?
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:0>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0model/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2)
'model/conv2d_transpose/conv2d_transpose?
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02/
-model/conv2d_transpose/BiasAdd/ReadVariableOp?
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:05model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2 
model/conv2d_transpose/BiasAdd?
model/conv2d_transpose/ReluRelu'model/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
model/conv2d_transpose/Relu?
model/conv2d_transpose_1/ShapeShape)model/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_1/Shape?
,model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_1/strided_slice/stack?
.model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_1/strided_slice/stack_1?
.model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_1/strided_slice/stack_2?
&model/conv2d_transpose_1/strided_sliceStridedSlice'model/conv2d_transpose_1/Shape:output:05model/conv2d_transpose_1/strided_slice/stack:output:07model/conv2d_transpose_1/strided_slice/stack_1:output:07model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_1/strided_slice?
 model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_1/stack/1?
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_1/stack/2?
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02"
 model/conv2d_transpose_1/stack/3?
model/conv2d_transpose_1/stackPack/model/conv2d_transpose_1/strided_slice:output:0)model/conv2d_transpose_1/stack/1:output:0)model/conv2d_transpose_1/stack/2:output:0)model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_1/stack?
.model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_1/strided_slice_1/stack?
0model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_1/strided_slice_1/stack_1?
0model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_1/strided_slice_1/stack_2?
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_1/strided_slice_1?
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:0@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0)model/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2+
)model/conv2d_transpose_1/conv2d_transpose?
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype021
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:07model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02"
 model/conv2d_transpose_1/BiasAdd?
model/conv2d_transpose_1/ReluRelu)model/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
model/conv2d_transpose_1/Relu?
model/conv2d_transpose_2/ShapeShape+model/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_2/Shape?
,model/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_2/strided_slice/stack?
.model/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_2/strided_slice/stack_1?
.model/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_2/strided_slice/stack_2?
&model/conv2d_transpose_2/strided_sliceStridedSlice'model/conv2d_transpose_2/Shape:output:05model/conv2d_transpose_2/strided_slice/stack:output:07model/conv2d_transpose_2/strided_slice/stack_1:output:07model/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_2/strided_slice?
 model/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_2/stack/1?
 model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_2/stack/2?
 model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2"
 model/conv2d_transpose_2/stack/3?
model/conv2d_transpose_2/stackPack/model/conv2d_transpose_2/strided_slice:output:0)model/conv2d_transpose_2/stack/1:output:0)model/conv2d_transpose_2/stack/2:output:0)model/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_2/stack?
.model/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_2/strided_slice_1/stack?
0model/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_2/strided_slice_1/stack_1?
0model/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_2/strided_slice_1/stack_2?
(model/conv2d_transpose_2/strided_slice_1StridedSlice'model/conv2d_transpose_2/stack:output:07model/conv2d_transpose_2/strided_slice_1/stack:output:09model/conv2d_transpose_2/strided_slice_1/stack_1:output:09model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_2/strided_slice_1?
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_2/stack:output:0@model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2+
)model/conv2d_transpose_2/conv2d_transpose?
/model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype021
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_2/BiasAddBiasAdd2model/conv2d_transpose_2/conv2d_transpose:output:07model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2"
 model/conv2d_transpose_2/BiasAdd?
model/conv2d_transpose_2/ReluRelu)model/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model/conv2d_transpose_2/Relu?
model/conv2d_transpose_3/ShapeShape+model/conv2d_transpose_2/Relu:activations:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_3/Shape?
,model/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_3/strided_slice/stack?
.model/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_3/strided_slice/stack_1?
.model/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_3/strided_slice/stack_2?
&model/conv2d_transpose_3/strided_sliceStridedSlice'model/conv2d_transpose_3/Shape:output:05model/conv2d_transpose_3/strided_slice/stack:output:07model/conv2d_transpose_3/strided_slice/stack_1:output:07model/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_3/strided_slice?
 model/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_3/stack/1?
 model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_3/stack/2?
 model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_3/stack/3?
model/conv2d_transpose_3/stackPack/model/conv2d_transpose_3/strided_slice:output:0)model/conv2d_transpose_3/stack/1:output:0)model/conv2d_transpose_3/stack/2:output:0)model/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_3/stack?
.model/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_3/strided_slice_1/stack?
0model/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_3/strided_slice_1/stack_1?
0model/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_3/strided_slice_1/stack_2?
(model/conv2d_transpose_3/strided_slice_1StridedSlice'model/conv2d_transpose_3/stack:output:07model/conv2d_transpose_3/strided_slice_1/stack:output:09model/conv2d_transpose_3/strided_slice_1/stack_1:output:09model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_3/strided_slice_1?
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_3/stack:output:0@model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_2/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)model/conv2d_transpose_3/conv2d_transpose?
/model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_3/BiasAddBiasAdd2model/conv2d_transpose_3/conv2d_transpose:output:07model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2"
 model/conv2d_transpose_3/BiasAdd?
model/conv2d_transpose_3/ReluRelu)model/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model/conv2d_transpose_3/Relu?
model/conv2d_transpose_4/ShapeShape+model/conv2d_transpose_3/Relu:activations:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_4/Shape?
,model/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_4/strided_slice/stack?
.model/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_4/strided_slice/stack_1?
.model/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_4/strided_slice/stack_2?
&model/conv2d_transpose_4/strided_sliceStridedSlice'model/conv2d_transpose_4/Shape:output:05model/conv2d_transpose_4/strided_slice/stack:output:07model/conv2d_transpose_4/strided_slice/stack_1:output:07model/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_4/strided_slice?
 model/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_4/stack/1?
 model/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_4/stack/2?
 model/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_4/stack/3?
model/conv2d_transpose_4/stackPack/model/conv2d_transpose_4/strided_slice:output:0)model/conv2d_transpose_4/stack/1:output:0)model/conv2d_transpose_4/stack/2:output:0)model/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_4/stack?
.model/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_4/strided_slice_1/stack?
0model/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_4/strided_slice_1/stack_1?
0model/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_4/strided_slice_1/stack_2?
(model/conv2d_transpose_4/strided_slice_1StridedSlice'model/conv2d_transpose_4/stack:output:07model/conv2d_transpose_4/strided_slice_1/stack:output:09model/conv2d_transpose_4/strided_slice_1/stack_1:output:09model/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_4/strided_slice_1?
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_4/stack:output:0@model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0+model/conv2d_transpose_3/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2+
)model/conv2d_transpose_4/conv2d_transpose?
/model/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model/conv2d_transpose_4/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_4/BiasAddBiasAdd2model/conv2d_transpose_4/conv2d_transpose:output:07model/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2"
 model/conv2d_transpose_4/BiasAdd?
 model/conv2d_transpose_4/SigmoidSigmoid)model/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 model/conv2d_transpose_4/Sigmoid?
IdentityIdentity$model/conv2d_transpose_4/Sigmoid:y:0.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_2/BiasAdd/ReadVariableOp9^model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_3/BiasAdd/ReadVariableOp9^model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_4/BiasAdd/ReadVariableOp9^model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2^
-model/conv2d_transpose/BiasAdd/ReadVariableOp-model/conv2d_transpose/BiasAdd/ReadVariableOp2p
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp6model/conv2d_transpose/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp/model/conv2d_transpose_1/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp/model/conv2d_transpose_2/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp/model/conv2d_transpose_3/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_4/BiasAdd/ReadVariableOp/model/conv2d_transpose_4/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_2"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????HN
conv2d_transpose_48
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?f
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
		variables

trainable_variables
regularization_losses
	keras_api

signatures
*^&call_and_return_all_conditional_losses
___call__
`_default_save_signature"?c
_tf_keras_network?b{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "name": "reshape", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_4", 0, 0]]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 72]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 72]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "name": "reshape", "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_4", "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]], "shared_object_id": 20}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_transpose_4", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?	

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"?
_tf_keras_layer?{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 5}
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*e&call_and_return_all_conditional_losses
f__call__"?

_tf_keras_layer?
{"name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["reshape", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 256]}}
?

kernel
bias
 	variables
!trainable_variables
"regularization_losses
#	keras_api
*g&call_and_return_all_conditional_losses
h__call__"?
_tf_keras_layer?
{"name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 72}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 72]}}
?

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*i&call_and_return_all_conditional_losses
j__call__"?
_tf_keras_layer?
{"name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 48]}}
?

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
*k&call_and_return_all_conditional_losses
l__call__"?
_tf_keras_layer?
{"name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
?

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
*m&call_and_return_all_conditional_losses
n__call__"?
_tf_keras_layer?
{"name": "conv2d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_4", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_3", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
v
0
1
2
3
4
5
$6
%7
*8
+9
010
111"
trackable_list_wrapper
v
0
1
2
3
4
5
$6
%7
*8
+9
010
111"
trackable_list_wrapper
J
o0
p1
q2
r3
s4
t5"
trackable_list_wrapper
?
		variables
6layer_metrics

trainable_variables
7non_trainable_variables
regularization_losses
8layer_regularization_losses
9metrics

:layers
___call__
`_default_save_signature
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,
userving_default"
signature_map
!:	H? 2dense_2/kernel
:? 2dense_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
?
	variables
;layer_metrics
trainable_variables
<non_trainable_variables
regularization_losses
=layer_regularization_losses
>metrics

?layers
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
@layer_metrics
trainable_variables
Anon_trainable_variables
regularization_losses
Blayer_regularization_losses
Cmetrics

Dlayers
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
2:0H?2conv2d_transpose/kernel
#:!H2conv2d_transpose/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
p0"
trackable_list_wrapper
?
	variables
Elayer_metrics
trainable_variables
Fnon_trainable_variables
regularization_losses
Glayer_regularization_losses
Hmetrics

Ilayers
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
3:10H2conv2d_transpose_1/kernel
%:#02conv2d_transpose_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
?
 	variables
Jlayer_metrics
!trainable_variables
Knon_trainable_variables
"regularization_losses
Llayer_regularization_losses
Mmetrics

Nlayers
h__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
3:1 02conv2d_transpose_2/kernel
%:# 2conv2d_transpose_2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
?
&	variables
Olayer_metrics
'trainable_variables
Pnon_trainable_variables
(regularization_losses
Qlayer_regularization_losses
Rmetrics

Slayers
j__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
'
s0"
trackable_list_wrapper
?
,	variables
Tlayer_metrics
-trainable_variables
Unon_trainable_variables
.regularization_losses
Vlayer_regularization_losses
Wmetrics

Xlayers
l__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
3:12conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
'
t0"
trackable_list_wrapper
?
2	variables
Ylayer_metrics
3trainable_variables
Znon_trainable_variables
4regularization_losses
[layer_regularization_losses
\metrics

]layers
n__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
p0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
r0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
s0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
t0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
@__inference_model_layer_call_and_return_conditional_losses_86143
@__inference_model_layer_call_and_return_conditional_losses_86354
@__inference_model_layer_call_and_return_conditional_losses_85686
@__inference_model_layer_call_and_return_conditional_losses_85811?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_model_layer_call_fn_85333
%__inference_model_layer_call_fn_86383
%__inference_model_layer_call_fn_86412
%__inference_model_layer_call_fn_85561?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_84835?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_2?????????H
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_86453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_86462?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_reshape_layer_call_and_return_conditional_losses_86476?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_reshape_layer_call_fn_86481?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84885?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
0__inference_conv2d_transpose_layer_call_fn_84895?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_84945?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????H
?2?
2__inference_conv2d_transpose_1_layer_call_fn_84955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????H
?2?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_85005?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????0
?2?
2__inference_conv2d_transpose_2_layer_call_fn_85015?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????0
?2?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_85065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
2__inference_conv2d_transpose_3_layer_call_fn_85075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_85125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
2__inference_conv2d_transpose_4_layer_call_fn_85135?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
__inference_loss_fn_0_86576?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_86596?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_86616?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_86636?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_86656?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_86676?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
#__inference_signature_wrapper_85932input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_84835?$%*+010?-
&?#
!?
input_2?????????H
? "O?L
J
conv2d_transpose_44?1
conv2d_transpose_4??????????
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_84945?I?F
??<
:?7
inputs+???????????????????????????H
? "??<
5?2
0+???????????????????????????0
? ?
2__inference_conv2d_transpose_1_layer_call_fn_84955?I?F
??<
:?7
inputs+???????????????????????????H
? "2?/+???????????????????????????0?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_85005?$%I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_conv2d_transpose_2_layer_call_fn_85015?$%I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
M__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_85065?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_3_layer_call_fn_85075?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
M__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_85125?01I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_4_layer_call_fn_85135?01I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_84885?J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????H
? ?
0__inference_conv2d_transpose_layer_call_fn_84895?J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????H?
B__inference_dense_2_layer_call_and_return_conditional_losses_86453]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0?????????? 
? {
'__inference_dense_2_layer_call_fn_86462P/?,
%?"
 ?
inputs?????????H
? "??????????? :
__inference_loss_fn_0_86576?

? 
? "? :
__inference_loss_fn_1_86596?

? 
? "? :
__inference_loss_fn_2_86616?

? 
? "? :
__inference_loss_fn_3_86636$?

? 
? "? :
__inference_loss_fn_4_86656*?

? 
? "? :
__inference_loss_fn_5_866760?

? 
? "? ?
@__inference_model_layer_call_and_return_conditional_losses_85686?$%*+018?5
.?+
!?
input_2?????????H
p 

 
? "??<
5?2
0+???????????????????????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_85811?$%*+018?5
.?+
!?
input_2?????????H
p

 
? "??<
5?2
0+???????????????????????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_86143v$%*+017?4
-?*
 ?
inputs?????????H
p 

 
? "-?*
#? 
0?????????
? ?
@__inference_model_layer_call_and_return_conditional_losses_86354v$%*+017?4
-?*
 ?
inputs?????????H
p

 
? "-?*
#? 
0?????????
? ?
%__inference_model_layer_call_fn_85333|$%*+018?5
.?+
!?
input_2?????????H
p 

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_85561|$%*+018?5
.?+
!?
input_2?????????H
p

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_86383{$%*+017?4
-?*
 ?
inputs?????????H
p 

 
? "2?/+????????????????????????????
%__inference_model_layer_call_fn_86412{$%*+017?4
-?*
 ?
inputs?????????H
p

 
? "2?/+????????????????????????????
B__inference_reshape_layer_call_and_return_conditional_losses_86476b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
'__inference_reshape_layer_call_fn_86481U0?-
&?#
!?
inputs?????????? 
? "!????????????
#__inference_signature_wrapper_85932?$%*+01;?8
? 
1?.
,
input_2!?
input_2?????????H"O?L
J
conv2d_transpose_44?1
conv2d_transpose_4?????????