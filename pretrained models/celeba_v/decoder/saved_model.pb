ƥ
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
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	H? *
shared_namedense_5/kernel
r
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes
:	H? *
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:? *
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:? *
dtype0
?
conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:H?**
shared_nameconv2d_transpose_5/kernel
?
-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*'
_output_shapes
:H?*
dtype0
?
conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:H*
dtype0
?
conv2d_transpose_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0H**
shared_nameconv2d_transpose_6/kernel
?
-conv2d_transpose_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/kernel*&
_output_shapes
:0H*
dtype0
?
conv2d_transpose_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*(
shared_nameconv2d_transpose_6/bias

+conv2d_transpose_6/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_6/bias*
_output_shapes
:0*
dtype0
?
conv2d_transpose_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 0**
shared_nameconv2d_transpose_7/kernel
?
-conv2d_transpose_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/kernel*&
_output_shapes
: 0*
dtype0
?
conv2d_transpose_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameconv2d_transpose_7/bias

+conv2d_transpose_7/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_7/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameconv2d_transpose_8/kernel
?
-conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_8/bias

+conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_9/kernel
?
-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_9/bias

+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
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
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
h

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
h

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
 
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
?
6non_trainable_variables
7layer_regularization_losses

8layers
9metrics
:layer_metrics
	regularization_losses

trainable_variables
	variables
 
ZX
VARIABLE_VALUEdense_5/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_5/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
;non_trainable_variables
<layer_regularization_losses

=layers
>metrics
?layer_metrics
regularization_losses
trainable_variables
	variables
 
 
 
?
@non_trainable_variables
Alayer_regularization_losses

Blayers
Cmetrics
Dlayer_metrics
regularization_losses
trainable_variables
	variables
ec
VARIABLE_VALUEconv2d_transpose_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Enon_trainable_variables
Flayer_regularization_losses

Glayers
Hmetrics
Ilayer_metrics
regularization_losses
trainable_variables
	variables
ec
VARIABLE_VALUEconv2d_transpose_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Jnon_trainable_variables
Klayer_regularization_losses

Llayers
Mmetrics
Nlayer_metrics
 regularization_losses
!trainable_variables
"	variables
ec
VARIABLE_VALUEconv2d_transpose_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
?
Onon_trainable_variables
Player_regularization_losses

Qlayers
Rmetrics
Slayer_metrics
&regularization_losses
'trainable_variables
(	variables
ec
VARIABLE_VALUEconv2d_transpose_8/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_8/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

*0
+1

*0
+1
?
Tnon_trainable_variables
Ulayer_regularization_losses

Vlayers
Wmetrics
Xlayer_metrics
,regularization_losses
-trainable_variables
.	variables
ec
VARIABLE_VALUEconv2d_transpose_9/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_9/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

00
11

00
11
?
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
\metrics
]layer_metrics
2regularization_losses
3trainable_variables
4	variables
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
 
 
z
serving_default_input_4Placeholder*'
_output_shapes
:?????????H*
dtype0*
shape:?????????H
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_4dense_5/kerneldense_5/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/biasconv2d_transpose_9/kernelconv2d_transpose_9/bias*
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
GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_104898
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp-conv2d_transpose_6/kernel/Read/ReadVariableOp+conv2d_transpose_6/bias/Read/ReadVariableOp-conv2d_transpose_7/kernel/Read/ReadVariableOp+conv2d_transpose_7/bias/Read/ReadVariableOp-conv2d_transpose_8/kernel/Read/ReadVariableOp+conv2d_transpose_8/bias/Read/ReadVariableOp-conv2d_transpose_9/kernel/Read/ReadVariableOp+conv2d_transpose_9/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_105701
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_5/kerneldense_5/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_6/kernelconv2d_transpose_6/biasconv2d_transpose_7/kernelconv2d_transpose_7/biasconv2d_transpose_8/kernelconv2d_transpose_8/biasconv2d_transpose_9/kernelconv2d_transpose_9/bias*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_105747Ɉ
?
?
__inference_loss_fn_2_105582[
Aconv2d_transpose_6_kernel_regularizer_abs_readvariableop_resource:0H
identity??8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_6/kernel/Regularizer/add_1:z:09^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp
?
?
(__inference_dense_5_layer_call_fn_105402

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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1041342
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
?
?
3__inference_conv2d_transpose_5_layer_call_fn_103861

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1038512
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
?A
?
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_104091

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpD
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
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp*
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
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_9_layer_call_fn_104101

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_1040912
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
?
__inference_loss_fn_3_105602[
Aconv2d_transpose_7_kernel_regularizer_abs_readvariableop_resource: 0
identity??8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_7/kernel/Regularizer/add_1:z:09^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp
??
?
!__inference__wrapped_model_103801
input_4A
.model_1_dense_5_matmul_readvariableop_resource:	H? >
/model_1_dense_5_biasadd_readvariableop_resource:	? ^
Cmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H?H
:model_1_conv2d_transpose_5_biasadd_readvariableop_resource:H]
Cmodel_1_conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0HH
:model_1_conv2d_transpose_6_biasadd_readvariableop_resource:0]
Cmodel_1_conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0H
:model_1_conv2d_transpose_7_biasadd_readvariableop_resource: ]
Cmodel_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource: H
:model_1_conv2d_transpose_8_biasadd_readvariableop_resource:]
Cmodel_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:H
:model_1_conv2d_transpose_9_biasadd_readvariableop_resource:
identity??1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp?:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?&model_1/dense_5/BiasAdd/ReadVariableOp?%model_1/dense_5/MatMul/ReadVariableOp?
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp?
model_1/dense_5/MatMulMatMulinput_4-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model_1/dense_5/MatMul?
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp?
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
model_1/dense_5/BiasAdd?
model_1/dense_5/ReluRelu model_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
model_1/dense_5/Relu?
model_1/reshape_1/ShapeShape"model_1/dense_5/Relu:activations:0*
T0*
_output_shapes
:2
model_1/reshape_1/Shape?
%model_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_1/strided_slice/stack?
'model_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_1?
'model_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_1/strided_slice/stack_2?
model_1/reshape_1/strided_sliceStridedSlice model_1/reshape_1/Shape:output:0.model_1/reshape_1/strided_slice/stack:output:00model_1/reshape_1/strided_slice/stack_1:output:00model_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_1/strided_slice?
!model_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_1/Reshape/shape/1?
!model_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_1/Reshape/shape/2?
!model_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2#
!model_1/reshape_1/Reshape/shape/3?
model_1/reshape_1/Reshape/shapePack(model_1/reshape_1/strided_slice:output:0*model_1/reshape_1/Reshape/shape/1:output:0*model_1/reshape_1/Reshape/shape/2:output:0*model_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_1/Reshape/shape?
model_1/reshape_1/ReshapeReshape"model_1/dense_5/Relu:activations:0(model_1/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
model_1/reshape_1/Reshape?
 model_1/conv2d_transpose_5/ShapeShape"model_1/reshape_1/Reshape:output:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/Shape?
.model_1/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_5/strided_slice/stack?
0model_1/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_1?
0model_1/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_5/strided_slice/stack_2?
(model_1/conv2d_transpose_5/strided_sliceStridedSlice)model_1/conv2d_transpose_5/Shape:output:07model_1/conv2d_transpose_5/strided_slice/stack:output:09model_1/conv2d_transpose_5/strided_slice/stack_1:output:09model_1/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_5/strided_slice?
"model_1/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_5/stack/1?
"model_1/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_5/stack/2?
"model_1/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2$
"model_1/conv2d_transpose_5/stack/3?
 model_1/conv2d_transpose_5/stackPack1model_1/conv2d_transpose_5/strided_slice:output:0+model_1/conv2d_transpose_5/stack/1:output:0+model_1/conv2d_transpose_5/stack/2:output:0+model_1/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_5/stack?
0model_1/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_5/strided_slice_1/stack?
2model_1/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_1?
2model_1/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_5/strided_slice_1/stack_2?
*model_1/conv2d_transpose_5/strided_slice_1StridedSlice)model_1/conv2d_transpose_5/stack:output:09model_1/conv2d_transpose_5/strided_slice_1/stack:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_5/strided_slice_1?
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02<
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_5/stack:output:0Bmodel_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0"model_1/reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_5/conv2d_transpose?
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype023
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_5/BiasAddBiasAdd4model_1/conv2d_transpose_5/conv2d_transpose:output:09model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2$
"model_1/conv2d_transpose_5/BiasAdd?
model_1/conv2d_transpose_5/ReluRelu+model_1/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2!
model_1/conv2d_transpose_5/Relu?
 model_1/conv2d_transpose_6/ShapeShape-model_1/conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_6/Shape?
.model_1/conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_6/strided_slice/stack?
0model_1/conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_6/strided_slice/stack_1?
0model_1/conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_6/strided_slice/stack_2?
(model_1/conv2d_transpose_6/strided_sliceStridedSlice)model_1/conv2d_transpose_6/Shape:output:07model_1/conv2d_transpose_6/strided_slice/stack:output:09model_1/conv2d_transpose_6/strided_slice/stack_1:output:09model_1/conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_6/strided_slice?
"model_1/conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_6/stack/1?
"model_1/conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_6/stack/2?
"model_1/conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02$
"model_1/conv2d_transpose_6/stack/3?
 model_1/conv2d_transpose_6/stackPack1model_1/conv2d_transpose_6/strided_slice:output:0+model_1/conv2d_transpose_6/stack/1:output:0+model_1/conv2d_transpose_6/stack/2:output:0+model_1/conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_6/stack?
0model_1/conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_6/strided_slice_1/stack?
2model_1/conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_6/strided_slice_1/stack_1?
2model_1/conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_6/strided_slice_1/stack_2?
*model_1/conv2d_transpose_6/strided_slice_1StridedSlice)model_1/conv2d_transpose_6/stack:output:09model_1/conv2d_transpose_6/strided_slice_1/stack:output:0;model_1/conv2d_transpose_6/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_6/strided_slice_1?
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02<
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_6/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_6/stack:output:0Bmodel_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_6/conv2d_transpose?
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype023
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_6/BiasAddBiasAdd4model_1/conv2d_transpose_6/conv2d_transpose:output:09model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02$
"model_1/conv2d_transpose_6/BiasAdd?
model_1/conv2d_transpose_6/ReluRelu+model_1/conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02!
model_1/conv2d_transpose_6/Relu?
 model_1/conv2d_transpose_7/ShapeShape-model_1/conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_7/Shape?
.model_1/conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_7/strided_slice/stack?
0model_1/conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_7/strided_slice/stack_1?
0model_1/conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_7/strided_slice/stack_2?
(model_1/conv2d_transpose_7/strided_sliceStridedSlice)model_1/conv2d_transpose_7/Shape:output:07model_1/conv2d_transpose_7/strided_slice/stack:output:09model_1/conv2d_transpose_7/strided_slice/stack_1:output:09model_1/conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_7/strided_slice?
"model_1/conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_7/stack/1?
"model_1/conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_7/stack/2?
"model_1/conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"model_1/conv2d_transpose_7/stack/3?
 model_1/conv2d_transpose_7/stackPack1model_1/conv2d_transpose_7/strided_slice:output:0+model_1/conv2d_transpose_7/stack/1:output:0+model_1/conv2d_transpose_7/stack/2:output:0+model_1/conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_7/stack?
0model_1/conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_7/strided_slice_1/stack?
2model_1/conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_7/strided_slice_1/stack_1?
2model_1/conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_7/strided_slice_1/stack_2?
*model_1/conv2d_transpose_7/strided_slice_1StridedSlice)model_1/conv2d_transpose_7/stack:output:09model_1/conv2d_transpose_7/strided_slice_1/stack:output:0;model_1/conv2d_transpose_7/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_7/strided_slice_1?
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02<
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_7/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_7/stack:output:0Bmodel_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2-
+model_1/conv2d_transpose_7/conv2d_transpose?
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_7/BiasAddBiasAdd4model_1/conv2d_transpose_7/conv2d_transpose:output:09model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2$
"model_1/conv2d_transpose_7/BiasAdd?
model_1/conv2d_transpose_7/ReluRelu+model_1/conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2!
model_1/conv2d_transpose_7/Relu?
 model_1/conv2d_transpose_8/ShapeShape-model_1/conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_8/Shape?
.model_1/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_8/strided_slice/stack?
0model_1/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_8/strided_slice/stack_1?
0model_1/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_8/strided_slice/stack_2?
(model_1/conv2d_transpose_8/strided_sliceStridedSlice)model_1/conv2d_transpose_8/Shape:output:07model_1/conv2d_transpose_8/strided_slice/stack:output:09model_1/conv2d_transpose_8/strided_slice/stack_1:output:09model_1/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_8/strided_slice?
"model_1/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_8/stack/1?
"model_1/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_8/stack/2?
"model_1/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_8/stack/3?
 model_1/conv2d_transpose_8/stackPack1model_1/conv2d_transpose_8/strided_slice:output:0+model_1/conv2d_transpose_8/stack/1:output:0+model_1/conv2d_transpose_8/stack/2:output:0+model_1/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_8/stack?
0model_1/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_8/strided_slice_1/stack?
2model_1/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_8/strided_slice_1/stack_1?
2model_1/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_8/strided_slice_1/stack_2?
*model_1/conv2d_transpose_8/strided_slice_1StridedSlice)model_1/conv2d_transpose_8/stack:output:09model_1/conv2d_transpose_8/strided_slice_1/stack:output:0;model_1/conv2d_transpose_8/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_8/strided_slice_1?
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_8/stack:output:0Bmodel_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_8/conv2d_transpose?
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_8/BiasAddBiasAdd4model_1/conv2d_transpose_8/conv2d_transpose:output:09model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"model_1/conv2d_transpose_8/BiasAdd?
model_1/conv2d_transpose_8/ReluRelu+model_1/conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
model_1/conv2d_transpose_8/Relu?
 model_1/conv2d_transpose_9/ShapeShape-model_1/conv2d_transpose_8/Relu:activations:0*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_9/Shape?
.model_1/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model_1/conv2d_transpose_9/strided_slice/stack?
0model_1/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_9/strided_slice/stack_1?
0model_1/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model_1/conv2d_transpose_9/strided_slice/stack_2?
(model_1/conv2d_transpose_9/strided_sliceStridedSlice)model_1/conv2d_transpose_9/Shape:output:07model_1/conv2d_transpose_9/strided_slice/stack:output:09model_1/conv2d_transpose_9/strided_slice/stack_1:output:09model_1/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model_1/conv2d_transpose_9/strided_slice?
"model_1/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_9/stack/1?
"model_1/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_9/stack/2?
"model_1/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_1/conv2d_transpose_9/stack/3?
 model_1/conv2d_transpose_9/stackPack1model_1/conv2d_transpose_9/strided_slice:output:0+model_1/conv2d_transpose_9/stack/1:output:0+model_1/conv2d_transpose_9/stack/2:output:0+model_1/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 model_1/conv2d_transpose_9/stack?
0model_1/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_1/conv2d_transpose_9/strided_slice_1/stack?
2model_1/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_9/strided_slice_1/stack_1?
2model_1/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_1/conv2d_transpose_9/strided_slice_1/stack_2?
*model_1/conv2d_transpose_9/strided_slice_1StridedSlice)model_1/conv2d_transpose_9/stack:output:09model_1/conv2d_transpose_9/strided_slice_1/stack:output:0;model_1/conv2d_transpose_9/strided_slice_1/stack_1:output:0;model_1/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_1/conv2d_transpose_9/strided_slice_1?
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_1_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02<
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
+model_1/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)model_1/conv2d_transpose_9/stack:output:0Bmodel_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0-model_1/conv2d_transpose_8/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+model_1/conv2d_transpose_9/conv2d_transpose?
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:model_1_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp?
"model_1/conv2d_transpose_9/BiasAddBiasAdd4model_1/conv2d_transpose_9/conv2d_transpose:output:09model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"model_1/conv2d_transpose_9/BiasAdd?
"model_1/conv2d_transpose_9/SigmoidSigmoid+model_1/conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"model_1/conv2d_transpose_9/Sigmoid?
IdentityIdentity&model_1/conv2d_transpose_9/Sigmoid:y:02^model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2^model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp;^model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2f
1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_5/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_6/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_6/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_7/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_7/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_8/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2f
1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp1model_1/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:model_1/conv2d_transpose_9/conv2d_transpose/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_4
?8
?
"__inference__traced_restore_105747
file_prefix2
assignvariableop_dense_5_kernel:	H? .
assignvariableop_1_dense_5_bias:	? G
,assignvariableop_2_conv2d_transpose_5_kernel:H?8
*assignvariableop_3_conv2d_transpose_5_bias:HF
,assignvariableop_4_conv2d_transpose_6_kernel:0H8
*assignvariableop_5_conv2d_transpose_6_bias:0F
,assignvariableop_6_conv2d_transpose_7_kernel: 08
*assignvariableop_7_conv2d_transpose_7_bias: F
,assignvariableop_8_conv2d_transpose_8_kernel: 8
*assignvariableop_9_conv2d_transpose_8_bias:G
-assignvariableop_10_conv2d_transpose_9_kernel:9
+assignvariableop_11_conv2d_transpose_9_bias:
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
AssignVariableOpAssignVariableOpassignvariableop_dense_5_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_5_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_conv2d_transpose_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp*assignvariableop_3_conv2d_transpose_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp,assignvariableop_4_conv2d_transpose_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_conv2d_transpose_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp,assignvariableop_6_conv2d_transpose_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp*assignvariableop_7_conv2d_transpose_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_conv2d_transpose_8_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_transpose_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_conv2d_transpose_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp+assignvariableop_11_conv2d_transpose_9_biasIdentity_11:output:0"/device:CPU:0*
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
?
?
__inference_loss_fn_4_105622[
Aconv2d_transpose_8_kernel_regularizer_abs_readvariableop_resource: 
identity??8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_8_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_8/kernel/Regularizer/add_1:z:09^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp
?
?
(__inference_model_1_layer_call_fn_104927

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
GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1042722
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
?
(__inference_model_1_layer_call_fn_104527
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1044712
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
_user_specified_name	input_4
?
?
__inference_loss_fn_5_105642[
Aconv2d_transpose_9_kernel_regularizer_abs_readvariableop_resource:
identity??8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_9_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_9/kernel/Regularizer/add_1:z:09^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_105167

inputs9
&dense_5_matmul_readvariableop_resource:	H? 6
'dense_5_biasadd_readvariableop_resource:	? V
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H?@
2conv2d_transpose_5_biasadd_readvariableop_resource:HU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_6_biasadd_readvariableop_resource:0U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identity??)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_5/Relul
reshape_1/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_5/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape~
conv2d_transpose_5/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose_5/BiasAdd?
conv2d_transpose_5/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose_5/Relu?
conv2d_transpose_6/ShapeShape%conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_6/BiasAdd?
conv2d_transpose_6/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_6/Relu?
conv2d_transpose_7/ShapeShape%conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slicez
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/1z
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/BiasAdd?
conv2d_transpose_7/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/Relu?
conv2d_transpose_8/ShapeShape%conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slicez
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/1z
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/BiasAdd?
conv2d_transpose_8/ReluRelu#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/Relu?
conv2d_transpose_9/ShapeShape%conv2d_transpose_8/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slicez
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/1z
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_8/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/SigmoidSigmoid#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_9/Sigmoid?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?

IdentityIdentityconv2d_transpose_9/Sigmoid:y:0*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?"
?
C__inference_dense_5_layer_call_and_return_conditional_losses_105428

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
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
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
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
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_105447

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
C__inference_model_1_layer_call_and_return_conditional_losses_104777
input_4!
dense_5_104655:	H? 
dense_5_104657:	? 4
conv2d_transpose_5_104661:H?'
conv2d_transpose_5_104663:H3
conv2d_transpose_6_104666:0H'
conv2d_transpose_6_104668:03
conv2d_transpose_7_104671: 0'
conv2d_transpose_7_104673: 3
conv2d_transpose_8_104676: '
conv2d_transpose_8_104678:3
conv2d_transpose_9_104681:'
conv2d_transpose_9_104683:
identity??*conv2d_transpose_5/StatefulPartitionedCall?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_6/StatefulPartitionedCall?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_7/StatefulPartitionedCall?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_8/StatefulPartitionedCall?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_9/StatefulPartitionedCall?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_5_104655dense_5_104657*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1041342!
dense_5/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1041542
reshape_1/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_5_104661conv2d_transpose_5_104663*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1038512,
*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0conv2d_transpose_6_104666conv2d_transpose_6_104668*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_1039112,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_104671conv2d_transpose_7_104673*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_1039712,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_104676conv2d_transpose_8_104678*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_1040312,
*conv2d_transpose_8/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_104681conv2d_transpose_9_104683*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_1040912,
*conv2d_transpose_9/StatefulPartitionedCall?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_104655*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_104655*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_5_104661*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_5_104661*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_6_104666*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_6_104666*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_7_104671*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_7_104671*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_8_104676*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_8_104676*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_9_104681*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_9_104681*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0+^conv2d_transpose_5/StatefulPartitionedCall9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_6/StatefulPartitionedCall9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_7/StatefulPartitionedCall9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_8/StatefulPartitionedCall9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_9/StatefulPartitionedCall9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_4
?
?
3__inference_conv2d_transpose_7_layer_call_fn_103981

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_1039712
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
?A
?
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_104031

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpD
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
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp*
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
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_reshape_1_layer_call_fn_105433

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
GPU2*0J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1041542
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
?
a
E__inference_reshape_1_layer_call_and_return_conditional_losses_104154

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
?&
?
__inference__traced_save_105701
file_prefix-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop8
4savev2_conv2d_transpose_6_kernel_read_readvariableop6
2savev2_conv2d_transpose_6_bias_read_readvariableop8
4savev2_conv2d_transpose_7_kernel_read_readvariableop6
2savev2_conv2d_transpose_7_bias_read_readvariableop8
4savev2_conv2d_transpose_8_kernel_read_readvariableop6
2savev2_conv2d_transpose_8_bias_read_readvariableop8
4savev2_conv2d_transpose_9_kernel_read_readvariableop6
2savev2_conv2d_transpose_9_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop4savev2_conv2d_transpose_6_kernel_read_readvariableop2savev2_conv2d_transpose_6_bias_read_readvariableop4savev2_conv2d_transpose_7_kernel_read_readvariableop2savev2_conv2d_transpose_7_bias_read_readvariableop4savev2_conv2d_transpose_8_kernel_read_readvariableop2savev2_conv2d_transpose_8_bias_read_readvariableop4savev2_conv2d_transpose_9_kernel_read_readvariableop2savev2_conv2d_transpose_9_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_105378

inputs9
&dense_5_matmul_readvariableop_resource:	H? 6
'dense_5_biasadd_readvariableop_resource:	? V
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:H?@
2conv2d_transpose_5_biasadd_readvariableop_resource:HU
;conv2d_transpose_6_conv2d_transpose_readvariableop_resource:0H@
2conv2d_transpose_6_biasadd_readvariableop_resource:0U
;conv2d_transpose_7_conv2d_transpose_readvariableop_resource: 0@
2conv2d_transpose_7_biasadd_readvariableop_resource: U
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource: @
2conv2d_transpose_8_biasadd_readvariableop_resource:U
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_9_biasadd_readvariableop_resource:
identity??)conv2d_transpose_5/BiasAdd/ReadVariableOp?2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_6/BiasAdd/ReadVariableOp?2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_7/BiasAdd/ReadVariableOp?2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/BiasAdd/ReadVariableOp?dense_5/MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02
dense_5/MatMul/ReadVariableOp?
dense_5/MatMulMatMulinputs%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/MatMul?
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:? *
dtype02 
dense_5/BiasAdd/ReadVariableOp?
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????? 2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:?????????? 2
dense_5/Relul
reshape_1/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapedense_5/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape~
conv2d_transpose_5/ShapeShapereshape_1/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_5/Shape?
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_5/strided_slice/stack?
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_1?
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_5/strided_slice/stack_2?
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_5/strided_slicez
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/1z
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_5/stack/2z
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :H2
conv2d_transpose_5/stack/3?
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_5/stack?
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_5/strided_slice_1/stack?
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_1?
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_5/strided_slice_1/stack_2?
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_5/strided_slice_1?
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype024
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0reshape_1/Reshape:output:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2%
#conv2d_transpose_5/conv2d_transpose?
)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02+
)conv2d_transpose_5/BiasAdd/ReadVariableOp?
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose_5/BiasAdd?
conv2d_transpose_5/ReluRelu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_transpose_5/Relu?
conv2d_transpose_6/ShapeShape%conv2d_transpose_5/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_6/Shape?
&conv2d_transpose_6/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_6/strided_slice/stack?
(conv2d_transpose_6/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_1?
(conv2d_transpose_6/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_6/strided_slice/stack_2?
 conv2d_transpose_6/strided_sliceStridedSlice!conv2d_transpose_6/Shape:output:0/conv2d_transpose_6/strided_slice/stack:output:01conv2d_transpose_6/strided_slice/stack_1:output:01conv2d_transpose_6/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_6/strided_slicez
conv2d_transpose_6/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/1z
conv2d_transpose_6/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_6/stack/2z
conv2d_transpose_6/stack/3Const*
_output_shapes
: *
dtype0*
value	B :02
conv2d_transpose_6/stack/3?
conv2d_transpose_6/stackPack)conv2d_transpose_6/strided_slice:output:0#conv2d_transpose_6/stack/1:output:0#conv2d_transpose_6/stack/2:output:0#conv2d_transpose_6/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_6/stack?
(conv2d_transpose_6/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_6/strided_slice_1/stack?
*conv2d_transpose_6/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_1?
*conv2d_transpose_6/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_6/strided_slice_1/stack_2?
"conv2d_transpose_6/strided_slice_1StridedSlice!conv2d_transpose_6/stack:output:01conv2d_transpose_6/strided_slice_1/stack:output:03conv2d_transpose_6/strided_slice_1/stack_1:output:03conv2d_transpose_6/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_6/strided_slice_1?
2conv2d_transpose_6/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype024
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_6/conv2d_transposeConv2DBackpropInput!conv2d_transpose_6/stack:output:0:conv2d_transpose_6/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_5/Relu:activations:0*
T0*/
_output_shapes
:?????????0*
paddingSAME*
strides
2%
#conv2d_transpose_6/conv2d_transpose?
)conv2d_transpose_6/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype02+
)conv2d_transpose_6/BiasAdd/ReadVariableOp?
conv2d_transpose_6/BiasAddBiasAdd,conv2d_transpose_6/conv2d_transpose:output:01conv2d_transpose_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_6/BiasAdd?
conv2d_transpose_6/ReluRelu#conv2d_transpose_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????02
conv2d_transpose_6/Relu?
conv2d_transpose_7/ShapeShape%conv2d_transpose_6/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_7/Shape?
&conv2d_transpose_7/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_7/strided_slice/stack?
(conv2d_transpose_7/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_1?
(conv2d_transpose_7/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_7/strided_slice/stack_2?
 conv2d_transpose_7/strided_sliceStridedSlice!conv2d_transpose_7/Shape:output:0/conv2d_transpose_7/strided_slice/stack:output:01conv2d_transpose_7/strided_slice/stack_1:output:01conv2d_transpose_7/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_7/strided_slicez
conv2d_transpose_7/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/1z
conv2d_transpose_7/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_7/stack/2z
conv2d_transpose_7/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_7/stack/3?
conv2d_transpose_7/stackPack)conv2d_transpose_7/strided_slice:output:0#conv2d_transpose_7/stack/1:output:0#conv2d_transpose_7/stack/2:output:0#conv2d_transpose_7/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_7/stack?
(conv2d_transpose_7/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_7/strided_slice_1/stack?
*conv2d_transpose_7/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_1?
*conv2d_transpose_7/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_7/strided_slice_1/stack_2?
"conv2d_transpose_7/strided_slice_1StridedSlice!conv2d_transpose_7/stack:output:01conv2d_transpose_7/strided_slice_1/stack:output:03conv2d_transpose_7/strided_slice_1/stack_1:output:03conv2d_transpose_7/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_7/strided_slice_1?
2conv2d_transpose_7/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype024
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_7/conv2d_transposeConv2DBackpropInput!conv2d_transpose_7/stack:output:0:conv2d_transpose_7/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_6/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_7/conv2d_transpose?
)conv2d_transpose_7/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_7_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_7/BiasAdd/ReadVariableOp?
conv2d_transpose_7/BiasAddBiasAdd,conv2d_transpose_7/conv2d_transpose:output:01conv2d_transpose_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/BiasAdd?
conv2d_transpose_7/ReluRelu#conv2d_transpose_7/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_7/Relu?
conv2d_transpose_8/ShapeShape%conv2d_transpose_7/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_8/Shape?
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_8/strided_slice/stack?
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_1?
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_8/strided_slice/stack_2?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_8/strided_slicez
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/1z
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/2z
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_8/stack/3?
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_8/stack?
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_8/strided_slice_1/stack?
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_1?
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_8/strided_slice_1/stack_2?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_8/strided_slice_1?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_7/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_8/conv2d_transpose?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_8/BiasAdd/ReadVariableOp?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/BiasAdd?
conv2d_transpose_8/ReluRelu#conv2d_transpose_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_8/Relu?
conv2d_transpose_9/ShapeShape%conv2d_transpose_8/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_9/Shape?
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_9/strided_slice/stack?
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_1?
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_9/strided_slice/stack_2?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_9/strided_slicez
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/1z
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/2z
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_9/stack/3?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_9/stack?
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_9/strided_slice_1/stack?
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_1?
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_9/strided_slice_1/stack_2?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_9/strided_slice_1?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype024
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_8/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_9/conv2d_transpose?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_9/BiasAdd/ReadVariableOp?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_9/BiasAdd?
conv2d_transpose_9/SigmoidSigmoid#conv2d_transpose_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_9/Sigmoid?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_6_conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_7_conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?

IdentityIdentityconv2d_transpose_9/Sigmoid:y:0*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_6/BiasAdd/ReadVariableOp3^conv2d_transpose_6/conv2d_transpose/ReadVariableOp9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_7/BiasAdd/ReadVariableOp3^conv2d_transpose_7/conv2d_transpose/ReadVariableOp9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_6/BiasAdd/ReadVariableOp)conv2d_transpose_6/BiasAdd/ReadVariableOp2h
2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2conv2d_transpose_6/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_7/BiasAdd/ReadVariableOp)conv2d_transpose_7/BiasAdd/ReadVariableOp2h
2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2conv2d_transpose_7/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_104272

inputs!
dense_5_104135:	H? 
dense_5_104137:	? 4
conv2d_transpose_5_104156:H?'
conv2d_transpose_5_104158:H3
conv2d_transpose_6_104161:0H'
conv2d_transpose_6_104163:03
conv2d_transpose_7_104166: 0'
conv2d_transpose_7_104168: 3
conv2d_transpose_8_104171: '
conv2d_transpose_8_104173:3
conv2d_transpose_9_104176:'
conv2d_transpose_9_104178:
identity??*conv2d_transpose_5/StatefulPartitionedCall?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_6/StatefulPartitionedCall?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_7/StatefulPartitionedCall?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_8/StatefulPartitionedCall?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_9/StatefulPartitionedCall?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_104135dense_5_104137*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1041342!
dense_5/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1041542
reshape_1/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_5_104156conv2d_transpose_5_104158*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1038512,
*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0conv2d_transpose_6_104161conv2d_transpose_6_104163*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_1039112,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_104166conv2d_transpose_7_104168*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_1039712,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_104171conv2d_transpose_8_104173*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_1040312,
*conv2d_transpose_8/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_104176conv2d_transpose_9_104178*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_1040912,
*conv2d_transpose_9/StatefulPartitionedCall?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_104135*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_104135*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_5_104156*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_5_104156*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_6_104161*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_6_104161*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_7_104166*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_7_104166*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_8_104171*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_8_104171*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_9_104176*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_9_104176*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0+^conv2d_transpose_5/StatefulPartitionedCall9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_6/StatefulPartitionedCall9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_7/StatefulPartitionedCall9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_8/StatefulPartitionedCall9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_9/StatefulPartitionedCall9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?A
?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_103911

inputsB
(conv2d_transpose_readvariableop_resource:0H-
biasadd_readvariableop_resource:0
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpD
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
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp*
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
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_104471

inputs!
dense_5_104349:	H? 
dense_5_104351:	? 4
conv2d_transpose_5_104355:H?'
conv2d_transpose_5_104357:H3
conv2d_transpose_6_104360:0H'
conv2d_transpose_6_104362:03
conv2d_transpose_7_104365: 0'
conv2d_transpose_7_104367: 3
conv2d_transpose_8_104370: '
conv2d_transpose_8_104372:3
conv2d_transpose_9_104375:'
conv2d_transpose_9_104377:
identity??*conv2d_transpose_5/StatefulPartitionedCall?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_6/StatefulPartitionedCall?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_7/StatefulPartitionedCall?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_8/StatefulPartitionedCall?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_9/StatefulPartitionedCall?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinputsdense_5_104349dense_5_104351*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1041342!
dense_5/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1041542
reshape_1/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_5_104355conv2d_transpose_5_104357*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1038512,
*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0conv2d_transpose_6_104360conv2d_transpose_6_104362*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_1039112,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_104365conv2d_transpose_7_104367*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_1039712,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_104370conv2d_transpose_8_104372*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_1040312,
*conv2d_transpose_8/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_104375conv2d_transpose_9_104377*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_1040912,
*conv2d_transpose_9/StatefulPartitionedCall?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_104349*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_104349*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_5_104355*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_5_104355*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_6_104360*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_6_104360*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_7_104365*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_7_104365*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_8_104370*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_8_104370*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_9_104375*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_9_104375*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0+^conv2d_transpose_5/StatefulPartitionedCall9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_6/StatefulPartitionedCall9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_7/StatefulPartitionedCall9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_8/StatefulPartitionedCall9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_9/StatefulPartitionedCall9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?A
?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_103971

inputsB
(conv2d_transpose_readvariableop_resource: 0-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpD
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
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp*
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
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????0
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_6_layer_call_fn_103921

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_1039112
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
?
?
(__inference_model_1_layer_call_fn_104956

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
GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1044712
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
?A
?
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_103851

inputsC
(conv2d_transpose_readvariableop_resource:H?-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpD
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
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?"
?
C__inference_dense_5_layer_call_and_return_conditional_losses_104134

inputs1
matmul_readvariableop_resource:	H? .
biasadd_readvariableop_resource:	? 
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
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
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
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
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
(__inference_model_1_layer_call_fn_104299
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_1042722
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
_user_specified_name	input_4
?
?
3__inference_conv2d_transpose_8_layer_call_fn_104041

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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_1040312
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
??
?
C__inference_model_1_layer_call_and_return_conditional_losses_104652
input_4!
dense_5_104530:	H? 
dense_5_104532:	? 4
conv2d_transpose_5_104536:H?'
conv2d_transpose_5_104538:H3
conv2d_transpose_6_104541:0H'
conv2d_transpose_6_104543:03
conv2d_transpose_7_104546: 0'
conv2d_transpose_7_104548: 3
conv2d_transpose_8_104551: '
conv2d_transpose_8_104553:3
conv2d_transpose_9_104556:'
conv2d_transpose_9_104558:
identity??*conv2d_transpose_5/StatefulPartitionedCall?8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_6/StatefulPartitionedCall?8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_7/StatefulPartitionedCall?8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_8/StatefulPartitionedCall?8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?*conv2d_transpose_9/StatefulPartitionedCall?8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?dense_5/StatefulPartitionedCall?-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
dense_5/StatefulPartitionedCallStatefulPartitionedCallinput_4dense_5_104530dense_5_104532*
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
GPU2*0J 8? *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_1041342!
dense_5/StatefulPartitionedCall?
reshape_1/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *N
fIRG
E__inference_reshape_1_layer_call_and_return_conditional_losses_1041542
reshape_1/PartitionedCall?
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0conv2d_transpose_5_104536conv2d_transpose_5_104538*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_1038512,
*conv2d_transpose_5/StatefulPartitionedCall?
*conv2d_transpose_6/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_5/StatefulPartitionedCall:output:0conv2d_transpose_6_104541conv2d_transpose_6_104543*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_1039112,
*conv2d_transpose_6/StatefulPartitionedCall?
*conv2d_transpose_7/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_6/StatefulPartitionedCall:output:0conv2d_transpose_7_104546conv2d_transpose_7_104548*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_1039712,
*conv2d_transpose_7/StatefulPartitionedCall?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_7/StatefulPartitionedCall:output:0conv2d_transpose_8_104551conv2d_transpose_8_104553*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_1040312,
*conv2d_transpose_8/StatefulPartitionedCall?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0conv2d_transpose_9_104556conv2d_transpose_9_104558*
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
GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_1040912,
*conv2d_transpose_9/StatefulPartitionedCall?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpdense_5_104530*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_5_104530*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_5_104536*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_5_104536*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
+conv2d_transpose_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_6/kernel/Regularizer/Const?
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_6_104541*&
_output_shapes
:0H*
dtype02:
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_6/kernel/Regularizer/AbsAbs@conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2+
)conv2d_transpose_6/kernel/Regularizer/Abs?
-conv2d_transpose_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_1?
)conv2d_transpose_6/kernel/Regularizer/SumSum-conv2d_transpose_6/kernel/Regularizer/Abs:y:06conv2d_transpose_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/Sum?
+conv2d_transpose_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_6/kernel/Regularizer/mul/x?
)conv2d_transpose_6/kernel/Regularizer/mulMul4conv2d_transpose_6/kernel/Regularizer/mul/x:output:02conv2d_transpose_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/mul?
)conv2d_transpose_6/kernel/Regularizer/addAddV24conv2d_transpose_6/kernel/Regularizer/Const:output:0-conv2d_transpose_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_6/kernel/Regularizer/add?
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_6_104541*&
_output_shapes
:0H*
dtype02=
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_6/kernel/Regularizer/SquareSquareCconv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:0H2.
,conv2d_transpose_6/kernel/Regularizer/Square?
-conv2d_transpose_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_6/kernel/Regularizer/Const_2?
+conv2d_transpose_6/kernel/Regularizer/Sum_1Sum0conv2d_transpose_6/kernel/Regularizer/Square:y:06conv2d_transpose_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/Sum_1?
-conv2d_transpose_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_6/kernel/Regularizer/mul_1/x?
+conv2d_transpose_6/kernel/Regularizer/mul_1Mul6conv2d_transpose_6/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/mul_1?
+conv2d_transpose_6/kernel/Regularizer/add_1AddV2-conv2d_transpose_6/kernel/Regularizer/add:z:0/conv2d_transpose_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_6/kernel/Regularizer/add_1?
+conv2d_transpose_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_7/kernel/Regularizer/Const?
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_7_104546*&
_output_shapes
: 0*
dtype02:
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_7/kernel/Regularizer/AbsAbs@conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02+
)conv2d_transpose_7/kernel/Regularizer/Abs?
-conv2d_transpose_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_1?
)conv2d_transpose_7/kernel/Regularizer/SumSum-conv2d_transpose_7/kernel/Regularizer/Abs:y:06conv2d_transpose_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/Sum?
+conv2d_transpose_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_7/kernel/Regularizer/mul/x?
)conv2d_transpose_7/kernel/Regularizer/mulMul4conv2d_transpose_7/kernel/Regularizer/mul/x:output:02conv2d_transpose_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/mul?
)conv2d_transpose_7/kernel/Regularizer/addAddV24conv2d_transpose_7/kernel/Regularizer/Const:output:0-conv2d_transpose_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_7/kernel/Regularizer/add?
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_7_104546*&
_output_shapes
: 0*
dtype02=
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_7/kernel/Regularizer/SquareSquareCconv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 02.
,conv2d_transpose_7/kernel/Regularizer/Square?
-conv2d_transpose_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_7/kernel/Regularizer/Const_2?
+conv2d_transpose_7/kernel/Regularizer/Sum_1Sum0conv2d_transpose_7/kernel/Regularizer/Square:y:06conv2d_transpose_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/Sum_1?
-conv2d_transpose_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_7/kernel/Regularizer/mul_1/x?
+conv2d_transpose_7/kernel/Regularizer/mul_1Mul6conv2d_transpose_7/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/mul_1?
+conv2d_transpose_7/kernel/Regularizer/add_1AddV2-conv2d_transpose_7/kernel/Regularizer/add:z:0/conv2d_transpose_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_7/kernel/Regularizer/add_1?
+conv2d_transpose_8/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_8/kernel/Regularizer/Const?
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_8_104551*&
_output_shapes
: *
dtype02:
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_8/kernel/Regularizer/AbsAbs@conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Abs?
-conv2d_transpose_8/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_1?
)conv2d_transpose_8/kernel/Regularizer/SumSum-conv2d_transpose_8/kernel/Regularizer/Abs:y:06conv2d_transpose_8/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/Sum?
+conv2d_transpose_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_8/kernel/Regularizer/mul/x?
)conv2d_transpose_8/kernel/Regularizer/mulMul4conv2d_transpose_8/kernel/Regularizer/mul/x:output:02conv2d_transpose_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/mul?
)conv2d_transpose_8/kernel/Regularizer/addAddV24conv2d_transpose_8/kernel/Regularizer/Const:output:0-conv2d_transpose_8/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_8/kernel/Regularizer/add?
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_8_104551*&
_output_shapes
: *
dtype02=
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_8/kernel/Regularizer/SquareSquareCconv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2.
,conv2d_transpose_8/kernel/Regularizer/Square?
-conv2d_transpose_8/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_8/kernel/Regularizer/Const_2?
+conv2d_transpose_8/kernel/Regularizer/Sum_1Sum0conv2d_transpose_8/kernel/Regularizer/Square:y:06conv2d_transpose_8/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/Sum_1?
-conv2d_transpose_8/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_8/kernel/Regularizer/mul_1/x?
+conv2d_transpose_8/kernel/Regularizer/mul_1Mul6conv2d_transpose_8/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_8/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/mul_1?
+conv2d_transpose_8/kernel/Regularizer/add_1AddV2-conv2d_transpose_8/kernel/Regularizer/add:z:0/conv2d_transpose_8/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_8/kernel/Regularizer/add_1?
+conv2d_transpose_9/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_9/kernel/Regularizer/Const?
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_transpose_9_104556*&
_output_shapes
:*
dtype02:
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_9/kernel/Regularizer/AbsAbs@conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2+
)conv2d_transpose_9/kernel/Regularizer/Abs?
-conv2d_transpose_9/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_1?
)conv2d_transpose_9/kernel/Regularizer/SumSum-conv2d_transpose_9/kernel/Regularizer/Abs:y:06conv2d_transpose_9/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/Sum?
+conv2d_transpose_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_9/kernel/Regularizer/mul/x?
)conv2d_transpose_9/kernel/Regularizer/mulMul4conv2d_transpose_9/kernel/Regularizer/mul/x:output:02conv2d_transpose_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/mul?
)conv2d_transpose_9/kernel/Regularizer/addAddV24conv2d_transpose_9/kernel/Regularizer/Const:output:0-conv2d_transpose_9/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_9/kernel/Regularizer/add?
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_transpose_9_104556*&
_output_shapes
:*
dtype02=
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_9/kernel/Regularizer/SquareSquareCconv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2.
,conv2d_transpose_9/kernel/Regularizer/Square?
-conv2d_transpose_9/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_9/kernel/Regularizer/Const_2?
+conv2d_transpose_9/kernel/Regularizer/Sum_1Sum0conv2d_transpose_9/kernel/Regularizer/Square:y:06conv2d_transpose_9/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/Sum_1?
-conv2d_transpose_9/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_9/kernel/Regularizer/mul_1/x?
+conv2d_transpose_9/kernel/Regularizer/mul_1Mul6conv2d_transpose_9/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_9/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/mul_1?
+conv2d_transpose_9/kernel/Regularizer/add_1AddV2-conv2d_transpose_9/kernel/Regularizer/add:z:0/conv2d_transpose_9/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_9/kernel/Regularizer/add_1?
IdentityIdentity3conv2d_transpose_9/StatefulPartitionedCall:output:0+^conv2d_transpose_5/StatefulPartitionedCall9^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_6/StatefulPartitionedCall9^conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_7/StatefulPartitionedCall9^conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_8/StatefulPartitionedCall9^conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp+^conv2d_transpose_9/StatefulPartitionedCall9^conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp ^dense_5/StatefulPartitionedCall.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????H: : : : : : : : : : : : 2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_6/StatefulPartitionedCall*conv2d_transpose_6/StatefulPartitionedCall2t
8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_6/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_6/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_7/StatefulPartitionedCall*conv2d_transpose_7/StatefulPartitionedCall2t
8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_7/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_7/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2t
8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_8/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_8/kernel/Regularizer/Square/ReadVariableOp2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2t
8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_9/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_9/kernel/Regularizer/Square/ReadVariableOp2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????H
!
_user_specified_name	input_4
?
?
__inference_loss_fn_1_105562\
Aconv2d_transpose_5_kernel_regularizer_abs_readvariableop_resource:H?
identity??8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
+conv2d_transpose_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+conv2d_transpose_5/kernel/Regularizer/Const?
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpAconv2d_transpose_5_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype02:
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp?
)conv2d_transpose_5/kernel/Regularizer/AbsAbs@conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2+
)conv2d_transpose_5/kernel/Regularizer/Abs?
-conv2d_transpose_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_1?
)conv2d_transpose_5/kernel/Regularizer/SumSum-conv2d_transpose_5/kernel/Regularizer/Abs:y:06conv2d_transpose_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/Sum?
+conv2d_transpose_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+conv2d_transpose_5/kernel/Regularizer/mul/x?
)conv2d_transpose_5/kernel/Regularizer/mulMul4conv2d_transpose_5/kernel/Regularizer/mul/x:output:02conv2d_transpose_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/mul?
)conv2d_transpose_5/kernel/Regularizer/addAddV24conv2d_transpose_5/kernel/Regularizer/Const:output:0-conv2d_transpose_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2+
)conv2d_transpose_5/kernel/Regularizer/add?
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpAconv2d_transpose_5_kernel_regularizer_abs_readvariableop_resource*'
_output_shapes
:H?*
dtype02=
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp?
,conv2d_transpose_5/kernel/Regularizer/SquareSquareCconv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*'
_output_shapes
:H?2.
,conv2d_transpose_5/kernel/Regularizer/Square?
-conv2d_transpose_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2/
-conv2d_transpose_5/kernel/Regularizer/Const_2?
+conv2d_transpose_5/kernel/Regularizer/Sum_1Sum0conv2d_transpose_5/kernel/Regularizer/Square:y:06conv2d_transpose_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/Sum_1?
-conv2d_transpose_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2/
-conv2d_transpose_5/kernel/Regularizer/mul_1/x?
+conv2d_transpose_5/kernel/Regularizer/mul_1Mul6conv2d_transpose_5/kernel/Regularizer/mul_1/x:output:04conv2d_transpose_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/mul_1?
+conv2d_transpose_5/kernel/Regularizer/add_1AddV2-conv2d_transpose_5/kernel/Regularizer/add:z:0/conv2d_transpose_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2-
+conv2d_transpose_5/kernel/Regularizer/add_1?
IdentityIdentity/conv2d_transpose_5/kernel/Regularizer/add_1:z:09^conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp<^conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2t
8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp8conv2d_transpose_5/kernel/Regularizer/Abs/ReadVariableOp2z
;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp;conv2d_transpose_5/kernel/Regularizer/Square/ReadVariableOp
?
?
__inference_loss_fn_0_105542I
6dense_5_kernel_regularizer_abs_readvariableop_resource:	H? 
identity??-dense_5/kernel/Regularizer/Abs/ReadVariableOp?0dense_5/kernel/Regularizer/Square/ReadVariableOp?
 dense_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dense_5/kernel/Regularizer/Const?
-dense_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6dense_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype02/
-dense_5/kernel/Regularizer/Abs/ReadVariableOp?
dense_5/kernel/Regularizer/AbsAbs5dense_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2 
dense_5/kernel/Regularizer/Abs?
"dense_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_1?
dense_5/kernel/Regularizer/SumSum"dense_5/kernel/Regularizer/Abs:y:0+dense_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/Sum?
 dense_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2"
 dense_5/kernel/Regularizer/mul/x?
dense_5/kernel/Regularizer/mulMul)dense_5/kernel/Regularizer/mul/x:output:0'dense_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/mul?
dense_5/kernel/Regularizer/addAddV2)dense_5/kernel/Regularizer/Const:output:0"dense_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2 
dense_5/kernel/Regularizer/add?
0dense_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp6dense_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes
:	H? *
dtype022
0dense_5/kernel/Regularizer/Square/ReadVariableOp?
!dense_5/kernel/Regularizer/SquareSquare8dense_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	H? 2#
!dense_5/kernel/Regularizer/Square?
"dense_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2$
"dense_5/kernel/Regularizer/Const_2?
 dense_5/kernel/Regularizer/Sum_1Sum%dense_5/kernel/Regularizer/Square:y:0+dense_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/Sum_1?
"dense_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2$
"dense_5/kernel/Regularizer/mul_1/x?
 dense_5/kernel/Regularizer/mul_1Mul+dense_5/kernel/Regularizer/mul_1/x:output:0)dense_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/mul_1?
 dense_5/kernel/Regularizer/add_1AddV2"dense_5/kernel/Regularizer/add:z:0$dense_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2"
 dense_5/kernel/Regularizer/add_1?
IdentityIdentity$dense_5/kernel/Regularizer/add_1:z:0.^dense_5/kernel/Regularizer/Abs/ReadVariableOp1^dense_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2^
-dense_5/kernel/Regularizer/Abs/ReadVariableOp-dense_5/kernel/Regularizer/Abs/ReadVariableOp2d
0dense_5/kernel/Regularizer/Square/ReadVariableOp0dense_5/kernel/Regularizer/Square/ReadVariableOp
?

?
$__inference_signature_wrapper_104898
input_4
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
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1038012
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
_user_specified_name	input_4"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_40
serving_default_input_4:0?????????HN
conv2d_transpose_98
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
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
^_default_save_signature
___call__
*`&call_and_return_all_conditional_losses"?c
_tf_keras_network?c{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_4", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "name": "reshape_1", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_6", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_7", "inbound_nodes": [[["conv2d_transpose_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_8", "inbound_nodes": [[["conv2d_transpose_7", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_9", "inbound_nodes": [[["conv2d_transpose_8", 0, 0, {}]]]}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_transpose_9", 0, 0]]}, "shared_object_id": 21, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 72]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 72]}, "float32", "input_4"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "name": "reshape_1", "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_5", "inbound_nodes": [[["reshape_1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_6", "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_7", "inbound_nodes": [[["conv2d_transpose_6", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_8", "inbound_nodes": [[["conv2d_transpose_7", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_9", "inbound_nodes": [[["conv2d_transpose_8", 0, 0, {}]]], "shared_object_id": 20}], "input_layers": [["input_4", 0, 0]], "output_layers": [["conv2d_transpose_9", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 72]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 4096, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_4", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 72}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 72]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
c__call__
*d&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 256]}}, "inbound_nodes": [[["dense_5", 0, 0, {}]]], "shared_object_id": 5}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
e__call__
*f&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_5", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["reshape_1", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 256]}}
?

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
g__call__
*h&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "conv2d_transpose_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_6", "trainable": true, "dtype": "float32", "filters": 48, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_5", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 72}}, "shared_object_id": 25}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 72]}}
?

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
i__call__
*j&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "conv2d_transpose_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_7", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_6", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 48}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 48]}}
?

*kernel
+bias
,regularization_losses
-trainable_variables
.	variables
/	keras_api
k__call__
*l&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "conv2d_transpose_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_8", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_7", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 27}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
?

0kernel
1bias
2regularization_losses
3trainable_variables
4	variables
5	keras_api
m__call__
*n&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "conv2d_transpose_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_9", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 18}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 19}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_8", 0, 0, {}]]], "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
J
o0
p1
q2
r3
s4
t5"
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
?
6non_trainable_variables
7layer_regularization_losses

8layers
9metrics
:layer_metrics
	regularization_losses

trainable_variables
	variables
___call__
^_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
,
userving_default"
signature_map
!:	H? 2dense_5/kernel
:? 2dense_5/bias
'
o0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
;non_trainable_variables
<layer_regularization_losses

=layers
>metrics
?layer_metrics
regularization_losses
trainable_variables
	variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables
Alayer_regularization_losses

Blayers
Cmetrics
Dlayer_metrics
regularization_losses
trainable_variables
	variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
4:2H?2conv2d_transpose_5/kernel
%:#H2conv2d_transpose_5/bias
'
p0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Enon_trainable_variables
Flayer_regularization_losses

Glayers
Hmetrics
Ilayer_metrics
regularization_losses
trainable_variables
	variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
3:10H2conv2d_transpose_6/kernel
%:#02conv2d_transpose_6/bias
'
q0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Jnon_trainable_variables
Klayer_regularization_losses

Llayers
Mmetrics
Nlayer_metrics
 regularization_losses
!trainable_variables
"	variables
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
3:1 02conv2d_transpose_7/kernel
%:# 2conv2d_transpose_7/bias
'
r0"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Onon_trainable_variables
Player_regularization_losses

Qlayers
Rmetrics
Slayer_metrics
&regularization_losses
'trainable_variables
(	variables
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
3:1 2conv2d_transpose_8/kernel
%:#2conv2d_transpose_8/bias
'
s0"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
Tnon_trainable_variables
Ulayer_regularization_losses

Vlayers
Wmetrics
Xlayer_metrics
,regularization_losses
-trainable_variables
.	variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
3:12conv2d_transpose_9/kernel
%:#2conv2d_transpose_9/bias
'
t0"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
Ynon_trainable_variables
Zlayer_regularization_losses

[layers
\metrics
]layer_metrics
2regularization_losses
3trainable_variables
4	variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_dict_wrapper
?2?
!__inference__wrapped_model_103801?
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
input_4?????????H
?2?
(__inference_model_1_layer_call_fn_104299
(__inference_model_1_layer_call_fn_104927
(__inference_model_1_layer_call_fn_104956
(__inference_model_1_layer_call_fn_104527?
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
?2?
C__inference_model_1_layer_call_and_return_conditional_losses_105167
C__inference_model_1_layer_call_and_return_conditional_losses_105378
C__inference_model_1_layer_call_and_return_conditional_losses_104652
C__inference_model_1_layer_call_and_return_conditional_losses_104777?
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
(__inference_dense_5_layer_call_fn_105402?
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
C__inference_dense_5_layer_call_and_return_conditional_losses_105428?
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
*__inference_reshape_1_layer_call_fn_105433?
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
E__inference_reshape_1_layer_call_and_return_conditional_losses_105447?
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
3__inference_conv2d_transpose_5_layer_call_fn_103861?
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
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_103851?
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
3__inference_conv2d_transpose_6_layer_call_fn_103921?
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
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_103911?
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
3__inference_conv2d_transpose_7_layer_call_fn_103981?
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
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_103971?
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
3__inference_conv2d_transpose_8_layer_call_fn_104041?
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
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_104031?
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
3__inference_conv2d_transpose_9_layer_call_fn_104101?
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
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_104091?
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
__inference_loss_fn_0_105542?
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
__inference_loss_fn_1_105562?
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
__inference_loss_fn_2_105582?
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
__inference_loss_fn_3_105602?
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
__inference_loss_fn_4_105622?
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
__inference_loss_fn_5_105642?
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
$__inference_signature_wrapper_104898input_4"?
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
!__inference__wrapped_model_103801?$%*+010?-
&?#
!?
input_4?????????H
? "O?L
J
conv2d_transpose_94?1
conv2d_transpose_9??????????
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_103851?J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????H
? ?
3__inference_conv2d_transpose_5_layer_call_fn_103861?J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????H?
N__inference_conv2d_transpose_6_layer_call_and_return_conditional_losses_103911?I?F
??<
:?7
inputs+???????????????????????????H
? "??<
5?2
0+???????????????????????????0
? ?
3__inference_conv2d_transpose_6_layer_call_fn_103921?I?F
??<
:?7
inputs+???????????????????????????H
? "2?/+???????????????????????????0?
N__inference_conv2d_transpose_7_layer_call_and_return_conditional_losses_103971?$%I?F
??<
:?7
inputs+???????????????????????????0
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_7_layer_call_fn_103981?$%I?F
??<
:?7
inputs+???????????????????????????0
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_104031?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_8_layer_call_fn_104041?*+I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
N__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_104091?01I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_9_layer_call_fn_104101?01I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
C__inference_dense_5_layer_call_and_return_conditional_losses_105428]/?,
%?"
 ?
inputs?????????H
? "&?#
?
0?????????? 
? |
(__inference_dense_5_layer_call_fn_105402P/?,
%?"
 ?
inputs?????????H
? "??????????? ;
__inference_loss_fn_0_105542?

? 
? "? ;
__inference_loss_fn_1_105562?

? 
? "? ;
__inference_loss_fn_2_105582?

? 
? "? ;
__inference_loss_fn_3_105602$?

? 
? "? ;
__inference_loss_fn_4_105622*?

? 
? "? ;
__inference_loss_fn_5_1056420?

? 
? "? ?
C__inference_model_1_layer_call_and_return_conditional_losses_104652?$%*+018?5
.?+
!?
input_4?????????H
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_104777?$%*+018?5
.?+
!?
input_4?????????H
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_1_layer_call_and_return_conditional_losses_105167v$%*+017?4
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
C__inference_model_1_layer_call_and_return_conditional_losses_105378v$%*+017?4
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
(__inference_model_1_layer_call_fn_104299|$%*+018?5
.?+
!?
input_4?????????H
p 

 
? "2?/+????????????????????????????
(__inference_model_1_layer_call_fn_104527|$%*+018?5
.?+
!?
input_4?????????H
p

 
? "2?/+????????????????????????????
(__inference_model_1_layer_call_fn_104927{$%*+017?4
-?*
 ?
inputs?????????H
p 

 
? "2?/+????????????????????????????
(__inference_model_1_layer_call_fn_104956{$%*+017?4
-?*
 ?
inputs?????????H
p

 
? "2?/+????????????????????????????
E__inference_reshape_1_layer_call_and_return_conditional_losses_105447b0?-
&?#
!?
inputs?????????? 
? ".?+
$?!
0??????????
? ?
*__inference_reshape_1_layer_call_fn_105433U0?-
&?#
!?
inputs?????????? 
? "!????????????
$__inference_signature_wrapper_104898?$%*+01;?8
? 
1?.
,
input_4!?
input_4?????????H"O?L
J
conv2d_transpose_94?1
conv2d_transpose_9?????????