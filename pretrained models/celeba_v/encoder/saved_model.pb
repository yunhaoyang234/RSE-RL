??
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
Conv2D

input"T
filter"T
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
H
ShardedFilename
basename	
shard

num_shards
filename
@
Softplus
features"T
activations"T"
Ttype:
2
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
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
: *
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
: *
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:@*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@H*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:H*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:H*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:H*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:H*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:H*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
?	?*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:?*
dtype0
m
z/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*
shared_name
z/kernel
f
z/kernel/Read/ReadVariableOpReadVariableOpz/kernel*
_output_shapes
:	?H*
dtype0
d
z/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namez/bias
]
z/bias/Read/ReadVariableOpReadVariableOpz/bias*
_output_shapes
:H*
dtype0
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	?H*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:H*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?,B?, B?,
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
regularization_losses
trainable_variables
	variables
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?
(axis
	)gamma
*beta
+moving_mean
,moving_variance
-regularization_losses
.trainable_variables
/	variables
0	keras_api
R
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
 
v
0
1
2
3
4
5
"6
#7
)8
*9
510
611
;12
<13
A14
B15
?
0
1
2
3
4
5
"6
#7
)8
*9
+10
,11
512
613
;14
<15
A16
B17
?
Gnon_trainable_variables
Hlayer_regularization_losses

Ilayers
Jmetrics
Klayer_metrics
regularization_losses
trainable_variables
	variables
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Lnon_trainable_variables
Mlayer_regularization_losses

Nlayers
Ometrics
Player_metrics
regularization_losses
trainable_variables
	variables
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Qnon_trainable_variables
Rlayer_regularization_losses

Slayers
Tmetrics
Ulayer_metrics
regularization_losses
trainable_variables
	variables
[Y
VARIABLE_VALUEconv2d_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Vnon_trainable_variables
Wlayer_regularization_losses

Xlayers
Ymetrics
Zlayer_metrics
regularization_losses
trainable_variables
 	variables
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
[non_trainable_variables
\layer_regularization_losses

]layers
^metrics
_layer_metrics
$regularization_losses
%trainable_variables
&	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
+2
,3
?
`non_trainable_variables
alayer_regularization_losses

blayers
cmetrics
dlayer_metrics
-regularization_losses
.trainable_variables
/	variables
 
 
 
?
enon_trainable_variables
flayer_regularization_losses

glayers
hmetrics
ilayer_metrics
1regularization_losses
2trainable_variables
3	variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
?
jnon_trainable_variables
klayer_regularization_losses

llayers
mmetrics
nlayer_metrics
7regularization_losses
8trainable_variables
9	variables
TR
VARIABLE_VALUEz/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEz/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
?
onon_trainable_variables
player_regularization_losses

qlayers
rmetrics
slayer_metrics
=regularization_losses
>trainable_variables
?	variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
?
tnon_trainable_variables
ulayer_regularization_losses

vlayers
wmetrics
xlayer_metrics
Cregularization_losses
Dtrainable_variables
E	variables

+0
,1
 
F
0
1
2
3
4
5
6
7
	8

9
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

+0
,1
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
?
serving_default_input_3Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_3/kerneldense_3/biasdense_4/kerneldense_4/biasz/kernelz/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_102632
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_103528
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedense_3/kerneldense_3/biasz/kernelz/biasdense_4/kerneldense_4/bias*
Tin
2*
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
"__inference__traced_restore_103592??
?#
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_103076

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
__inference__traced_save_103528
file_prefix.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop
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
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
22
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
?: ::: : : @:@:@H:H:H:H:H:H:
?	?:?:	?H:H:	?H:H: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@H: 

_output_shapes
:H: 	

_output_shapes
:H: 


_output_shapes
:H: 

_output_shapes
:H: 

_output_shapes
:H:&"
 
_output_shapes
:
?	?:!

_output_shapes	
:?:%!

_output_shapes
:	?H: 

_output_shapes
:H:%!

_output_shapes
:	?H: 

_output_shapes
:H:

_output_shapes
: 
?
?
(__inference_encoder_layer_call_fn_101948
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1019072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103282

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
)__inference_conv2d_7_layer_call_fn_103150

inputs!
unknown:@H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1017542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103300

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_103410Q
7conv2d_5_kernel_regularizer_abs_readvariableop_resource: 
identity??.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_5_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_5/kernel/Regularizer/add_1:z:0/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_101521

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_103430Q
7conv2d_6_kernel_regularizer_abs_readvariableop_resource: @
identity??.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_6_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_6/kernel/Regularizer/add_1:z:0/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp
?n
?
!__inference__wrapped_model_101499
input_3I
/encoder_conv2d_4_conv2d_readvariableop_resource:>
0encoder_conv2d_4_biasadd_readvariableop_resource:I
/encoder_conv2d_5_conv2d_readvariableop_resource: >
0encoder_conv2d_5_biasadd_readvariableop_resource: I
/encoder_conv2d_6_conv2d_readvariableop_resource: @>
0encoder_conv2d_6_biasadd_readvariableop_resource:@I
/encoder_conv2d_7_conv2d_readvariableop_resource:@H>
0encoder_conv2d_7_biasadd_readvariableop_resource:HC
5encoder_batch_normalization_1_readvariableop_resource:HE
7encoder_batch_normalization_1_readvariableop_1_resource:HT
Fencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HV
Hencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:HB
.encoder_dense_3_matmul_readvariableop_resource:
?	?>
/encoder_dense_3_biasadd_readvariableop_resource:	?A
.encoder_dense_4_matmul_readvariableop_resource:	?H=
/encoder_dense_4_biasadd_readvariableop_resource:H;
(encoder_z_matmul_readvariableop_resource:	?H7
)encoder_z_biasadd_readvariableop_resource:H
identity

identity_1??=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp??encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?,encoder/batch_normalization_1/ReadVariableOp?.encoder/batch_normalization_1/ReadVariableOp_1?'encoder/conv2d_4/BiasAdd/ReadVariableOp?&encoder/conv2d_4/Conv2D/ReadVariableOp?'encoder/conv2d_5/BiasAdd/ReadVariableOp?&encoder/conv2d_5/Conv2D/ReadVariableOp?'encoder/conv2d_6/BiasAdd/ReadVariableOp?&encoder/conv2d_6/Conv2D/ReadVariableOp?'encoder/conv2d_7/BiasAdd/ReadVariableOp?&encoder/conv2d_7/Conv2D/ReadVariableOp?&encoder/dense_3/BiasAdd/ReadVariableOp?%encoder/dense_3/MatMul/ReadVariableOp?&encoder/dense_4/BiasAdd/ReadVariableOp?%encoder/dense_4/MatMul/ReadVariableOp? encoder/z/BiasAdd/ReadVariableOp?encoder/z/MatMul/ReadVariableOp?
&encoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&encoder/conv2d_4/Conv2D/ReadVariableOp?
encoder/conv2d_4/Conv2DConv2Dinput_3.encoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder/conv2d_4/Conv2D?
'encoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv2d_4/BiasAdd/ReadVariableOp?
encoder/conv2d_4/BiasAddBiasAdd encoder/conv2d_4/Conv2D:output:0/encoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_4/BiasAdd?
encoder/conv2d_4/ReluRelu!encoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d_4/Relu?
&encoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_5/Conv2D/ReadVariableOp?
encoder/conv2d_5/Conv2DConv2D#encoder/conv2d_4/Relu:activations:0.encoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
encoder/conv2d_5/Conv2D?
'encoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_5/BiasAdd/ReadVariableOp?
encoder/conv2d_5/BiasAddBiasAdd encoder/conv2d_5/Conv2D:output:0/encoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_5/BiasAdd?
encoder/conv2d_5/ReluRelu!encoder/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_5/Relu?
&encoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_6/Conv2D/ReadVariableOp?
encoder/conv2d_6/Conv2DConv2D#encoder/conv2d_5/Relu:activations:0.encoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
encoder/conv2d_6/Conv2D?
'encoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_6/BiasAdd/ReadVariableOp?
encoder/conv2d_6/BiasAddBiasAdd encoder/conv2d_6/Conv2D:output:0/encoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_6/BiasAdd?
encoder/conv2d_6/ReluRelu!encoder/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_6/Relu?
&encoder/conv2d_7/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02(
&encoder/conv2d_7/Conv2D/ReadVariableOp?
encoder/conv2d_7/Conv2DConv2D#encoder/conv2d_6/Relu:activations:0.encoder/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
encoder/conv2d_7/Conv2D?
'encoder/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'encoder/conv2d_7/BiasAdd/ReadVariableOp?
encoder/conv2d_7/BiasAddBiasAdd encoder/conv2d_7/Conv2D:output:0/encoder/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
encoder/conv2d_7/BiasAdd?
encoder/conv2d_7/ReluRelu!encoder/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
encoder/conv2d_7/Relu?
,encoder/batch_normalization_1/ReadVariableOpReadVariableOp5encoder_batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype02.
,encoder/batch_normalization_1/ReadVariableOp?
.encoder/batch_normalization_1/ReadVariableOp_1ReadVariableOp7encoder_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype020
.encoder/batch_normalization_1/ReadVariableOp_1?
=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpFencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02?
=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpHencoder_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02A
?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
.encoder/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#encoder/conv2d_7/Relu:activations:04encoder/batch_normalization_1/ReadVariableOp:value:06encoder/batch_normalization_1/ReadVariableOp_1:value:0Eencoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Gencoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 20
.encoder/batch_normalization_1/FusedBatchNormV3?
encoder/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
encoder/flatten_1/Const?
encoder/flatten_1/ReshapeReshape2encoder/batch_normalization_1/FusedBatchNormV3:y:0 encoder/flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????	2
encoder/flatten_1/Reshape?
%encoder/dense_3/MatMul/ReadVariableOpReadVariableOp.encoder_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02'
%encoder/dense_3/MatMul/ReadVariableOp?
encoder/dense_3/MatMulMatMul"encoder/flatten_1/Reshape:output:0-encoder/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/dense_3/MatMul?
&encoder/dense_3/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&encoder/dense_3/BiasAdd/ReadVariableOp?
encoder/dense_3/BiasAddBiasAdd encoder/dense_3/MatMul:product:0.encoder/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/dense_3/BiasAdd?
encoder/dense_3/ReluRelu encoder/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
encoder/dense_3/Relu?
%encoder/dense_4/MatMul/ReadVariableOpReadVariableOp.encoder_dense_4_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02'
%encoder/dense_4/MatMul/ReadVariableOp?
encoder/dense_4/MatMulMatMul"encoder/dense_3/Relu:activations:0-encoder/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_4/MatMul?
&encoder/dense_4/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02(
&encoder/dense_4/BiasAdd/ReadVariableOp?
encoder/dense_4/BiasAddBiasAdd encoder/dense_4/MatMul:product:0.encoder/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_4/BiasAdd?
encoder/dense_4/SoftplusSoftplus encoder/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_4/Softplus?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul"encoder/dense_3/Relu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/z/MatMul?
 encoder/z/BiasAdd/ReadVariableOpReadVariableOp)encoder_z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02"
 encoder/z/BiasAdd/ReadVariableOp?
encoder/z/BiasAddBiasAddencoder/z/MatMul:product:0(encoder/z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/z/BiasAdd?
IdentityIdentity&encoder/dense_4/Softplus:activations:0>^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^encoder/batch_normalization_1/ReadVariableOp/^encoder/batch_normalization_1/ReadVariableOp_1(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp'^encoder/dense_3/BiasAdd/ReadVariableOp&^encoder/dense_3/MatMul/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identityencoder/z/BiasAdd:output:0>^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@^encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1-^encoder/batch_normalization_1/ReadVariableOp/^encoder/batch_normalization_1/ReadVariableOp_1(^encoder/conv2d_4/BiasAdd/ReadVariableOp'^encoder/conv2d_4/Conv2D/ReadVariableOp(^encoder/conv2d_5/BiasAdd/ReadVariableOp'^encoder/conv2d_5/Conv2D/ReadVariableOp(^encoder/conv2d_6/BiasAdd/ReadVariableOp'^encoder/conv2d_6/Conv2D/ReadVariableOp(^encoder/conv2d_7/BiasAdd/ReadVariableOp'^encoder/conv2d_7/Conv2D/ReadVariableOp'^encoder/dense_3/BiasAdd/ReadVariableOp&^encoder/dense_3/MatMul/ReadVariableOp'^encoder/dense_4/BiasAdd/ReadVariableOp&^encoder/dense_4/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2~
=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp=encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?encoder/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12\
,encoder/batch_normalization_1/ReadVariableOp,encoder/batch_normalization_1/ReadVariableOp2`
.encoder/batch_normalization_1/ReadVariableOp_1.encoder/batch_normalization_1/ReadVariableOp_12R
'encoder/conv2d_4/BiasAdd/ReadVariableOp'encoder/conv2d_4/BiasAdd/ReadVariableOp2P
&encoder/conv2d_4/Conv2D/ReadVariableOp&encoder/conv2d_4/Conv2D/ReadVariableOp2R
'encoder/conv2d_5/BiasAdd/ReadVariableOp'encoder/conv2d_5/BiasAdd/ReadVariableOp2P
&encoder/conv2d_5/Conv2D/ReadVariableOp&encoder/conv2d_5/Conv2D/ReadVariableOp2R
'encoder/conv2d_6/BiasAdd/ReadVariableOp'encoder/conv2d_6/BiasAdd/ReadVariableOp2P
&encoder/conv2d_6/Conv2D/ReadVariableOp&encoder/conv2d_6/Conv2D/ReadVariableOp2R
'encoder/conv2d_7/BiasAdd/ReadVariableOp'encoder/conv2d_7/BiasAdd/ReadVariableOp2P
&encoder/conv2d_7/Conv2D/ReadVariableOp&encoder/conv2d_7/Conv2D/ReadVariableOp2P
&encoder/dense_3/BiasAdd/ReadVariableOp&encoder/dense_3/BiasAdd/ReadVariableOp2N
%encoder/dense_3/MatMul/ReadVariableOp%encoder/dense_3/MatMul/ReadVariableOp2P
&encoder/dense_4/BiasAdd/ReadVariableOp&encoder/dense_4/BiasAdd/ReadVariableOp2N
%encoder/dense_4/MatMul/ReadVariableOp%encoder/dense_4/MatMul/ReadVariableOp2D
 encoder/z/BiasAdd/ReadVariableOp encoder/z/BiasAdd/ReadVariableOp2B
encoder/z/MatMul/ReadVariableOpencoder/z/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?	
?
=__inference_z_layer_call_and_return_conditional_losses_101839

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_101565

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
՘
?
C__inference_encoder_layer_call_and_return_conditional_losses_102417
input_3)
conv2d_4_102310:
conv2d_4_102312:)
conv2d_5_102315: 
conv2d_5_102317: )
conv2d_6_102320: @
conv2d_6_102322:@)
conv2d_7_102325:@H
conv2d_7_102327:H*
batch_normalization_1_102330:H*
batch_normalization_1_102332:H*
batch_normalization_1_102334:H*
batch_normalization_1_102336:H"
dense_3_102340:
?	?
dense_3_102342:	?!
dense_4_102345:	?H
dense_4_102347:H
z_102350:	?H
z_102352:H
identity

identity_1??-batch_normalization_1/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp? conv2d_5/StatefulPartitionedCall?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp? conv2d_6/StatefulPartitionedCall?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp? conv2d_7/StatefulPartitionedCall?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_4_102310conv2d_4_102312*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1016582"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_102315conv2d_5_102317*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1016902"
 conv2d_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_102320conv2d_6_102322*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1017222"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_102325conv2d_7_102327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1017542"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_102330batch_normalization_1_102332batch_normalization_1_102334batch_normalization_1_102336*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1017772/
-batch_normalization_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1017932
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_102340dense_3_102342*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1018062!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102345dense_4_102347*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1018232!
dense_4/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0z_102350z_102352*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1018392
z/StatefulPartitionedCall?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_102310*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_102310*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_102315*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_102315*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_102320*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_102320*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_102325*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_102325*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_4/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
)__inference_conv2d_6_layer_call_fn_103100

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1017222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?O
?
"__inference__traced_restore_103592
file_prefix:
 assignvariableop_conv2d_4_kernel:.
 assignvariableop_1_conv2d_4_bias:<
"assignvariableop_2_conv2d_5_kernel: .
 assignvariableop_3_conv2d_5_bias: <
"assignvariableop_4_conv2d_6_kernel: @.
 assignvariableop_5_conv2d_6_bias:@<
"assignvariableop_6_conv2d_7_kernel:@H.
 assignvariableop_7_conv2d_7_bias:H<
.assignvariableop_8_batch_normalization_1_gamma:H;
-assignvariableop_9_batch_normalization_1_beta:HC
5assignvariableop_10_batch_normalization_1_moving_mean:HG
9assignvariableop_11_batch_normalization_1_moving_variance:H6
"assignvariableop_12_dense_3_kernel:
?	?/
 assignvariableop_13_dense_3_bias:	?/
assignvariableop_14_z_kernel:	?H(
assignvariableop_15_z_bias:H5
"assignvariableop_16_dense_4_kernel:	?H.
 assignvariableop_17_dense_4_bias:H
identity_19??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_conv2d_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_z_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_z_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_4_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_4_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_179
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_18?
Identity_19IdentityIdentity_18:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_19"#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
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
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_101793

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?

?
C__inference_dense_4_layer_call_and_return_conditional_losses_103370

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2

Softplus?
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103264

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_102632
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1014992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
6__inference_batch_normalization_1_layer_call_fn_103228

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1020172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
(__inference_dense_4_layer_call_fn_103359

inputs
unknown:	?H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1018232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_103311

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_103331

inputs2
matmul_readvariableop_resource:
?	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_103202

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
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
GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1015652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_102675

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1019072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_103026

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ә
?
C__inference_encoder_layer_call_and_return_conditional_losses_102527
input_3)
conv2d_4_102420:
conv2d_4_102422:)
conv2d_5_102425: 
conv2d_5_102427: )
conv2d_6_102430: @
conv2d_6_102432:@)
conv2d_7_102435:@H
conv2d_7_102437:H*
batch_normalization_1_102440:H*
batch_normalization_1_102442:H*
batch_normalization_1_102444:H*
batch_normalization_1_102446:H"
dense_3_102450:
?	?
dense_3_102452:	?!
dense_4_102455:	?H
dense_4_102457:H
z_102460:	?H
z_102462:H
identity

identity_1??-batch_normalization_1/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp? conv2d_5/StatefulPartitionedCall?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp? conv2d_6/StatefulPartitionedCall?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp? conv2d_7/StatefulPartitionedCall?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_4_102420conv2d_4_102422*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1016582"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_102425conv2d_5_102427*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1016902"
 conv2d_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_102430conv2d_6_102432*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1017222"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_102435conv2d_7_102437*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1017542"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_102440batch_normalization_1_102442batch_normalization_1_102444batch_normalization_1_102446*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1020172/
-batch_normalization_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1017932
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_102450dense_3_102452*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1018062!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102455dense_4_102457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1018232!
dense_4/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0z_102460z_102462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1018392
z/StatefulPartitionedCall?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_102420*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_102420*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_102425*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_102425*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_102430*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_102430*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_102435*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_102435*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_4/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?#
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_103176

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
Relu?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
D__inference_conv2d_5_layer_call_and_return_conditional_losses_101690

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_5_layer_call_fn_103050

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1016902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_102847

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@H6
(conv2d_7_biasadd_readvariableop_resource:H;
-batch_normalization_1_readvariableop_resource:H=
/batch_normalization_1_readvariableop_1_resource:HL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:H:
&dense_3_matmul_readvariableop_resource:
?	?6
'dense_3_biasadd_readvariableop_resource:	?9
&dense_4_matmul_readvariableop_resource:	?H5
'dense_4_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_5/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_6/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_7/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshape*batch_normalization_1/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_4/BiasAdd|
dense_4/SoftplusSoftplusdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_4/Softplus?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense_3/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2

z/MatMul?
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
z/BiasAdd/ReadVariableOp?
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	z/BiasAdd?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentityz/BiasAdd:output:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?	

Identity_1Identitydense_4/Softplus:activations:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103246

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_102976

inputsA
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:A
'conv2d_5_conv2d_readvariableop_resource: 6
(conv2d_5_biasadd_readvariableop_resource: A
'conv2d_6_conv2d_readvariableop_resource: @6
(conv2d_6_biasadd_readvariableop_resource:@A
'conv2d_7_conv2d_readvariableop_resource:@H6
(conv2d_7_biasadd_readvariableop_resource:H;
-batch_normalization_1_readvariableop_resource:H=
/batch_normalization_1_readvariableop_1_resource:HL
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:HN
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:H:
&dense_3_matmul_readvariableop_resource:
?	?6
'dense_3_biasadd_readvariableop_resource:	?9
&dense_4_matmul_readvariableop_resource:	?H5
'dense_4_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2Dconv2d_4/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_5/BiasAdd{
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_5/Relu?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2Dconv2d_5/Relu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_6/BiasAdd{
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_6/Relu?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2Dconv2d_6/Relu:activations:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_7/Relu?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:H*
dtype02&
$batch_normalization_1/ReadVariableOp?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:H*
dtype02(
&batch_normalization_1/ReadVariableOp_1?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_7/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2(
&batch_normalization_1/FusedBatchNormV3?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1s
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten_1/Const?
flatten_1/ReshapeReshape*batch_normalization_1/FusedBatchNormV3:y:0flatten_1/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten_1/Reshape?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMulflatten_1/Reshape:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_3/Relu?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_4/BiasAdd|
dense_4/SoftplusSoftplusdense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_4/Softplus?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense_3/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2

z/MatMul?
z/BiasAdd/ReadVariableOpReadVariableOp!z_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
z/BiasAdd/ReadVariableOp?
	z/BiasAddBiasAddz/MatMul:product:0 z/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
	z/BiasAdd?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?	
IdentityIdentityz/BiasAdd:output:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?	

Identity_1Identitydense_4/Softplus:activations:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_z_layer_call_fn_103340

inputs
unknown:	?H
	unknown_0:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1018392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_103215

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1017772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
И
?
C__inference_encoder_layer_call_and_return_conditional_losses_102223

inputs)
conv2d_4_102116:
conv2d_4_102118:)
conv2d_5_102121: 
conv2d_5_102123: )
conv2d_6_102126: @
conv2d_6_102128:@)
conv2d_7_102131:@H
conv2d_7_102133:H*
batch_normalization_1_102136:H*
batch_normalization_1_102138:H*
batch_normalization_1_102140:H*
batch_normalization_1_102142:H"
dense_3_102146:
?	?
dense_3_102148:	?!
dense_4_102151:	?H
dense_4_102153:H
z_102156:	?H
z_102158:H
identity

identity_1??-batch_normalization_1/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp? conv2d_5/StatefulPartitionedCall?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp? conv2d_6/StatefulPartitionedCall?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp? conv2d_7/StatefulPartitionedCall?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_102116conv2d_4_102118*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1016582"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_102121conv2d_5_102123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1016902"
 conv2d_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_102126conv2d_6_102128*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1017222"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_102131conv2d_7_102133*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1017542"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_102136batch_normalization_1_102138batch_normalization_1_102140batch_normalization_1_102142*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1020172/
-batch_normalization_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1017932
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_102146dense_3_102148*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1018062!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_102151dense_4_102153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1018232!
dense_4/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0z_102156z_102158*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1018392
z/StatefulPartitionedCall?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_102116*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_102116*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_102121*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_102121*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_102126*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_102126*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_102131*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_102131*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_4/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_103390Q
7conv2d_4_kernel_regularizer_abs_readvariableop_resource:
identity??.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_4_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_4/kernel/Regularizer/add_1:z:0/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp
?
?
)__inference_conv2d_4_layer_call_fn_103000

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1016582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
=__inference_z_layer_call_and_return_conditional_losses_103350

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_102307
input_3!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1022232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_102017

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_101777

inputs%
readvariableop_resource:H'
readvariableop_1_resource:H6
(fusedbatchnormv3_readvariableop_resource:H8
*fusedbatchnormv3_readvariableop_1_resource:H
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:H*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:H*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????H: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
F
*__inference_flatten_1_layer_call_fn_103305

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1017932
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????H:W S
/
_output_shapes
:?????????H
 
_user_specified_nameinputs
?
?
6__inference_batch_normalization_1_layer_call_fn_103189

inputs
unknown:H
	unknown_0:H
	unknown_1:H
	unknown_2:H
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1015212
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????H: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????H
 
_user_specified_nameinputs
?#
?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_101754

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
Relu?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?#
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_101722

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
C__inference_dense_3_layer_call_and_return_conditional_losses_101806

inputs2
matmul_readvariableop_resource:
?	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?#
?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_103126

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
(__inference_dense_3_layer_call_fn_103320

inputs
unknown:
?	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1018062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????	
 
_user_specified_nameinputs
?

?
C__inference_dense_4_layer_call_and_return_conditional_losses_101823

inputs1
matmul_readvariableop_resource:	?H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:H*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2	
BiasAddd
SoftplusSoftplusBiasAdd:output:0*
T0*'
_output_shapes
:?????????H2

Softplus?
IdentityIdentitySoftplus:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_encoder_layer_call_fn_102718

inputs!
unknown:
	unknown_0:#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@H
	unknown_6:H
	unknown_7:H
	unknown_8:H
	unknown_9:H

unknown_10:H

unknown_11:
?	?

unknown_12:	?

unknown_13:	?H

unknown_14:H

unknown_15:	?H

unknown_16:H
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????H:?????????H*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1022232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_3_103450Q
7conv2d_7_kernel_regularizer_abs_readvariableop_resource:@H
identity??.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_7_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_7/kernel/Regularizer/add_1:z:0/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp
?#
?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_101658

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ҙ
?
C__inference_encoder_layer_call_and_return_conditional_losses_101907

inputs)
conv2d_4_101659:
conv2d_4_101661:)
conv2d_5_101691: 
conv2d_5_101693: )
conv2d_6_101723: @
conv2d_6_101725:@)
conv2d_7_101755:@H
conv2d_7_101757:H*
batch_normalization_1_101778:H*
batch_normalization_1_101780:H*
batch_normalization_1_101782:H*
batch_normalization_1_101784:H"
dense_3_101807:
?	?
dense_3_101809:	?!
dense_4_101824:	?H
dense_4_101826:H
z_101840:	?H
z_101842:H
identity

identity_1??-batch_normalization_1/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall?.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_4/kernel/Regularizer/Square/ReadVariableOp? conv2d_5/StatefulPartitionedCall?.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_5/kernel/Regularizer/Square/ReadVariableOp? conv2d_6/StatefulPartitionedCall?.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_6/kernel/Regularizer/Square/ReadVariableOp? conv2d_7/StatefulPartitionedCall?.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?z/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_101659conv2d_4_101661*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_1016582"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_101691conv2d_5_101693*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_1016902"
 conv2d_5/StatefulPartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_101723conv2d_6_101725*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_1017222"
 conv2d_6/StatefulPartitionedCall?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0conv2d_7_101755conv2d_7_101757*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_1017542"
 conv2d_7/StatefulPartitionedCall?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batch_normalization_1_101778batch_normalization_1_101780batch_normalization_1_101782batch_normalization_1_101784*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????H*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1017772/
-batch_normalization_1/StatefulPartitionedCall?
flatten_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_1017932
flatten_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_3_101807dense_3_101809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_1018062!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_101824dense_4_101826*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_1018232!
dense_4/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0z_101840z_101842*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????H*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *F
fAR?
=__inference_z_layer_call_and_return_conditional_losses_1018392
z/StatefulPartitionedCall?
!conv2d_4/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_4/kernel/Regularizer/Const?
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_4_101659*&
_output_shapes
:*
dtype020
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_4/kernel/Regularizer/AbsAbs6conv2d_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2!
conv2d_4/kernel/Regularizer/Abs?
#conv2d_4/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_1?
conv2d_4/kernel/Regularizer/SumSum#conv2d_4/kernel/Regularizer/Abs:y:0,conv2d_4/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/Sum?
!conv2d_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_4/kernel/Regularizer/mul/x?
conv2d_4/kernel/Regularizer/mulMul*conv2d_4/kernel/Regularizer/mul/x:output:0(conv2d_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/mul?
conv2d_4/kernel/Regularizer/addAddV2*conv2d_4/kernel/Regularizer/Const:output:0#conv2d_4/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_4/kernel/Regularizer/add?
1conv2d_4/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_4_101659*&
_output_shapes
:*
dtype023
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_4/kernel/Regularizer/SquareSquare9conv2d_4/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2$
"conv2d_4/kernel/Regularizer/Square?
#conv2d_4/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_4/kernel/Regularizer/Const_2?
!conv2d_4/kernel/Regularizer/Sum_1Sum&conv2d_4/kernel/Regularizer/Square:y:0,conv2d_4/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/Sum_1?
#conv2d_4/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_4/kernel/Regularizer/mul_1/x?
!conv2d_4/kernel/Regularizer/mul_1Mul,conv2d_4/kernel/Regularizer/mul_1/x:output:0*conv2d_4/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/mul_1?
!conv2d_4/kernel/Regularizer/add_1AddV2#conv2d_4/kernel/Regularizer/add:z:0%conv2d_4/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_4/kernel/Regularizer/add_1?
!conv2d_5/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_5/kernel/Regularizer/Const?
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_5_101691*&
_output_shapes
: *
dtype020
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_5/kernel/Regularizer/AbsAbs6conv2d_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Abs?
#conv2d_5/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_1?
conv2d_5/kernel/Regularizer/SumSum#conv2d_5/kernel/Regularizer/Abs:y:0,conv2d_5/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/Sum?
!conv2d_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_5/kernel/Regularizer/mul/x?
conv2d_5/kernel/Regularizer/mulMul*conv2d_5/kernel/Regularizer/mul/x:output:0(conv2d_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/mul?
conv2d_5/kernel/Regularizer/addAddV2*conv2d_5/kernel/Regularizer/Const:output:0#conv2d_5/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_5/kernel/Regularizer/add?
1conv2d_5/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_5_101691*&
_output_shapes
: *
dtype023
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_5/kernel/Regularizer/SquareSquare9conv2d_5/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_5/kernel/Regularizer/Square?
#conv2d_5/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_5/kernel/Regularizer/Const_2?
!conv2d_5/kernel/Regularizer/Sum_1Sum&conv2d_5/kernel/Regularizer/Square:y:0,conv2d_5/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/Sum_1?
#conv2d_5/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_5/kernel/Regularizer/mul_1/x?
!conv2d_5/kernel/Regularizer/mul_1Mul,conv2d_5/kernel/Regularizer/mul_1/x:output:0*conv2d_5/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/mul_1?
!conv2d_5/kernel/Regularizer/add_1AddV2#conv2d_5/kernel/Regularizer/add:z:0%conv2d_5/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_5/kernel/Regularizer/add_1?
!conv2d_6/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_6/kernel/Regularizer/Const?
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_6_101723*&
_output_shapes
: @*
dtype020
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_6/kernel/Regularizer/AbsAbs6conv2d_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_6/kernel/Regularizer/Abs?
#conv2d_6/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_1?
conv2d_6/kernel/Regularizer/SumSum#conv2d_6/kernel/Regularizer/Abs:y:0,conv2d_6/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/Sum?
!conv2d_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_6/kernel/Regularizer/mul/x?
conv2d_6/kernel/Regularizer/mulMul*conv2d_6/kernel/Regularizer/mul/x:output:0(conv2d_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/mul?
conv2d_6/kernel/Regularizer/addAddV2*conv2d_6/kernel/Regularizer/Const:output:0#conv2d_6/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_6/kernel/Regularizer/add?
1conv2d_6/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_6_101723*&
_output_shapes
: @*
dtype023
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_6/kernel/Regularizer/SquareSquare9conv2d_6/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_6/kernel/Regularizer/Square?
#conv2d_6/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_6/kernel/Regularizer/Const_2?
!conv2d_6/kernel/Regularizer/Sum_1Sum&conv2d_6/kernel/Regularizer/Square:y:0,conv2d_6/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/Sum_1?
#conv2d_6/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_6/kernel/Regularizer/mul_1/x?
!conv2d_6/kernel/Regularizer/mul_1Mul,conv2d_6/kernel/Regularizer/mul_1/x:output:0*conv2d_6/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/mul_1?
!conv2d_6/kernel/Regularizer/add_1AddV2#conv2d_6/kernel/Regularizer/add:z:0%conv2d_6/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_6/kernel/Regularizer/add_1?
!conv2d_7/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_7/kernel/Regularizer/Const?
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_7_101755*&
_output_shapes
:@H*
dtype020
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_7/kernel/Regularizer/AbsAbs6conv2d_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_7/kernel/Regularizer/Abs?
#conv2d_7/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_1?
conv2d_7/kernel/Regularizer/SumSum#conv2d_7/kernel/Regularizer/Abs:y:0,conv2d_7/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/Sum?
!conv2d_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_7/kernel/Regularizer/mul/x?
conv2d_7/kernel/Regularizer/mulMul*conv2d_7/kernel/Regularizer/mul/x:output:0(conv2d_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/mul?
conv2d_7/kernel/Regularizer/addAddV2*conv2d_7/kernel/Regularizer/Const:output:0#conv2d_7/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_7/kernel/Regularizer/add?
1conv2d_7/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_7_101755*&
_output_shapes
:@H*
dtype023
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_7/kernel/Regularizer/SquareSquare9conv2d_7/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_7/kernel/Regularizer/Square?
#conv2d_7/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_7/kernel/Regularizer/Const_2?
!conv2d_7/kernel/Regularizer/Sum_1Sum&conv2d_7/kernel/Regularizer/Square:y:0,conv2d_7/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/Sum_1?
#conv2d_7/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_7/kernel/Regularizer/mul_1/x?
!conv2d_7/kernel/Regularizer/mul_1Mul,conv2d_7/kernel/Regularizer/mul_1/x:output:0*conv2d_7/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/mul_1?
!conv2d_7/kernel/Regularizer/add_1AddV2#conv2d_7/kernel/Regularizer/add:z:0%conv2d_7/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_7/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_4/StatefulPartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall/^conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_4/kernel/Regularizer/Square/ReadVariableOp!^conv2d_5/StatefulPartitionedCall/^conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_5/kernel/Regularizer/Square/ReadVariableOp!^conv2d_6/StatefulPartitionedCall/^conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_6/kernel/Regularizer/Square/ReadVariableOp!^conv2d_7/StatefulPartitionedCall/^conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_7/kernel/Regularizer/Square/ReadVariableOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2`
.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp.conv2d_4/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_4/kernel/Regularizer/Square/ReadVariableOp1conv2d_4/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2`
.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp.conv2d_5/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_5/kernel/Regularizer/Square/ReadVariableOp1conv2d_5/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2`
.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp.conv2d_6/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_6/kernel/Regularizer/Square/ReadVariableOp1conv2d_6/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2`
.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp.conv2d_7/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_7/kernel/Regularizer/Square/ReadVariableOp1conv2d_7/kernel/Regularizer/Square/ReadVariableOp2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_38
serving_default_input_3:0?????????;
dense_40
StatefulPartitionedCall:0?????????H5
z0
StatefulPartitionedCall:1?????????Htensorflow/serving/predict:??
?n
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
regularization_losses
trainable_variables
	variables
	keras_api

signatures
y_default_save_signature
z__call__
*{&call_and_return_all_conditional_losses"?j
_tf_keras_network?i{"name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["z", 0, 0], ["dense_4", 0, 0]]}, "shared_object_id": 29, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 16, 16, 1]}, "float32", "input_3"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_7", "inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["conv2d_7", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 28}], "input_layers": [["input_3", 0, 0]], "output_layers": [["z", 0, 0], ["dense_4", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_3", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 1]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
~__call__
*&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
?

kernel
bias
regularization_losses
trainable_variables
 	variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
?

"kernel
#bias
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 64]}}
?

(axis
	)gamma
*beta
+moving_mean
,moving_variance
-regularization_losses
.trainable_variables
/	variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_7", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 72}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 72]}}
?
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 36}}
?	

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten_1", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?	

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_3", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
@
?0
?1
?2
?3"
trackable_list_wrapper
?
0
1
2
3
4
5
"6
#7
)8
*9
510
611
;12
<13
A14
B15"
trackable_list_wrapper
?
0
1
2
3
4
5
"6
#7
)8
*9
+10
,11
512
613
;14
<15
A16
B17"
trackable_list_wrapper
?
Gnon_trainable_variables
Hlayer_regularization_losses

Ilayers
Jmetrics
Klayer_metrics
regularization_losses
trainable_variables
	variables
z__call__
y_default_save_signature
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):'2conv2d_4/kernel
:2conv2d_4/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Lnon_trainable_variables
Mlayer_regularization_losses

Nlayers
Ometrics
Player_metrics
regularization_losses
trainable_variables
	variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_5/kernel
: 2conv2d_5/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Qnon_trainable_variables
Rlayer_regularization_losses

Slayers
Tmetrics
Ulayer_metrics
regularization_losses
trainable_variables
	variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_6/kernel
:@2conv2d_6/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Vnon_trainable_variables
Wlayer_regularization_losses

Xlayers
Ymetrics
Zlayer_metrics
regularization_losses
trainable_variables
 	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
):'@H2conv2d_7/kernel
:H2conv2d_7/bias
(
?0"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
[non_trainable_variables
\layer_regularization_losses

]layers
^metrics
_layer_metrics
$regularization_losses
%trainable_variables
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'H2batch_normalization_1/gamma
(:&H2batch_normalization_1/beta
1:/H (2!batch_normalization_1/moving_mean
5:3H (2%batch_normalization_1/moving_variance
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
<
)0
*1
+2
,3"
trackable_list_wrapper
?
`non_trainable_variables
alayer_regularization_losses

blayers
cmetrics
dlayer_metrics
-regularization_losses
.trainable_variables
/	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
enon_trainable_variables
flayer_regularization_losses

glayers
hmetrics
ilayer_metrics
1regularization_losses
2trainable_variables
3	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
?	?2dense_3/kernel
:?2dense_3/bias
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
jnon_trainable_variables
klayer_regularization_losses

llayers
mmetrics
nlayer_metrics
7regularization_losses
8trainable_variables
9	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?H2z/kernel
:H2z/bias
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
onon_trainable_variables
player_regularization_losses

qlayers
rmetrics
slayer_metrics
=regularization_losses
>trainable_variables
?	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	?H2dense_4/kernel
:H2dense_4/bias
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
tnon_trainable_variables
ulayer_regularization_losses

vlayers
wmetrics
xlayer_metrics
Cregularization_losses
Dtrainable_variables
E	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
+0
,1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
!__inference__wrapped_model_101499?
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
annotations? *.?+
)?&
input_3?????????
?2?
(__inference_encoder_layer_call_fn_101948
(__inference_encoder_layer_call_fn_102675
(__inference_encoder_layer_call_fn_102718
(__inference_encoder_layer_call_fn_102307?
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
C__inference_encoder_layer_call_and_return_conditional_losses_102847
C__inference_encoder_layer_call_and_return_conditional_losses_102976
C__inference_encoder_layer_call_and_return_conditional_losses_102417
C__inference_encoder_layer_call_and_return_conditional_losses_102527?
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
)__inference_conv2d_4_layer_call_fn_103000?
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
D__inference_conv2d_4_layer_call_and_return_conditional_losses_103026?
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
)__inference_conv2d_5_layer_call_fn_103050?
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
D__inference_conv2d_5_layer_call_and_return_conditional_losses_103076?
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
)__inference_conv2d_6_layer_call_fn_103100?
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
D__inference_conv2d_6_layer_call_and_return_conditional_losses_103126?
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
)__inference_conv2d_7_layer_call_fn_103150?
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
D__inference_conv2d_7_layer_call_and_return_conditional_losses_103176?
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
?2?
6__inference_batch_normalization_1_layer_call_fn_103189
6__inference_batch_normalization_1_layer_call_fn_103202
6__inference_batch_normalization_1_layer_call_fn_103215
6__inference_batch_normalization_1_layer_call_fn_103228?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103246
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103264
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103282
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103300?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
*__inference_flatten_1_layer_call_fn_103305?
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
E__inference_flatten_1_layer_call_and_return_conditional_losses_103311?
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
(__inference_dense_3_layer_call_fn_103320?
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
C__inference_dense_3_layer_call_and_return_conditional_losses_103331?
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
"__inference_z_layer_call_fn_103340?
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
=__inference_z_layer_call_and_return_conditional_losses_103350?
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
(__inference_dense_4_layer_call_fn_103359?
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
C__inference_dense_4_layer_call_and_return_conditional_losses_103370?
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
__inference_loss_fn_0_103390?
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
__inference_loss_fn_1_103410?
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
__inference_loss_fn_2_103430?
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
__inference_loss_fn_3_103450?
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
$__inference_signature_wrapper_102632input_3"?
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
!__inference__wrapped_model_101499?"#)*+,56AB;<8?5
.?+
)?&
input_3?????????
? "S?P
,
dense_4!?
dense_4?????????H
 
z?
z?????????H?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103246?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p 
? "??<
5?2
0+???????????????????????????H
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103264?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p
? "??<
5?2
0+???????????????????????????H
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103282r)*+,;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
Q__inference_batch_normalization_1_layer_call_and_return_conditional_losses_103300r)*+,;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
6__inference_batch_normalization_1_layer_call_fn_103189?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p 
? "2?/+???????????????????????????H?
6__inference_batch_normalization_1_layer_call_fn_103202?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p
? "2?/+???????????????????????????H?
6__inference_batch_normalization_1_layer_call_fn_103215e)*+,;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
6__inference_batch_normalization_1_layer_call_fn_103228e)*+,;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
D__inference_conv2d_4_layer_call_and_return_conditional_losses_103026l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_4_layer_call_fn_103000_7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2d_5_layer_call_and_return_conditional_losses_103076l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
)__inference_conv2d_5_layer_call_fn_103050_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
D__inference_conv2d_6_layer_call_and_return_conditional_losses_103126l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
)__inference_conv2d_6_layer_call_fn_103100_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
D__inference_conv2d_7_layer_call_and_return_conditional_losses_103176l"#7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????H
? ?
)__inference_conv2d_7_layer_call_fn_103150_"#7?4
-?*
(?%
inputs?????????@
? " ??????????H?
C__inference_dense_3_layer_call_and_return_conditional_losses_103331^560?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? }
(__inference_dense_3_layer_call_fn_103320Q560?-
&?#
!?
inputs??????????	
? "????????????
C__inference_dense_4_layer_call_and_return_conditional_losses_103370]AB0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? |
(__inference_dense_4_layer_call_fn_103359PAB0?-
&?#
!?
inputs??????????
? "??????????H?
C__inference_encoder_layer_call_and_return_conditional_losses_102417?"#)*+,56AB;<@?=
6?3
)?&
input_3?????????
p 

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_102527?"#)*+,56AB;<@?=
6?3
)?&
input_3?????????
p

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_102847?"#)*+,56AB;<??<
5?2
(?%
inputs?????????
p 

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_102976?"#)*+,56AB;<??<
5?2
(?%
inputs?????????
p

 
? "K?H
A?>
?
0/0?????????H
?
0/1?????????H
? ?
(__inference_encoder_layer_call_fn_101948?"#)*+,56AB;<@?=
6?3
)?&
input_3?????????
p 

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_102307?"#)*+,56AB;<@?=
6?3
)?&
input_3?????????
p

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_102675?"#)*+,56AB;<??<
5?2
(?%
inputs?????????
p 

 
? "=?:
?
0?????????H
?
1?????????H?
(__inference_encoder_layer_call_fn_102718?"#)*+,56AB;<??<
5?2
(?%
inputs?????????
p

 
? "=?:
?
0?????????H
?
1?????????H?
E__inference_flatten_1_layer_call_and_return_conditional_losses_103311a7?4
-?*
(?%
inputs?????????H
? "&?#
?
0??????????	
? ?
*__inference_flatten_1_layer_call_fn_103305T7?4
-?*
(?%
inputs?????????H
? "???????????	;
__inference_loss_fn_0_103390?

? 
? "? ;
__inference_loss_fn_1_103410?

? 
? "? ;
__inference_loss_fn_2_103430?

? 
? "? ;
__inference_loss_fn_3_103450"?

? 
? "? ?
$__inference_signature_wrapper_102632?"#)*+,56AB;<C?@
? 
9?6
4
input_3)?&
input_3?????????"S?P
,
dense_4!?
dense_4?????????H
 
z?
z?????????H?
=__inference_z_layer_call_and_return_conditional_losses_103350];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? v
"__inference_z_layer_call_fn_103340P;<0?-
&?#
!?
inputs??????????
? "??????????H