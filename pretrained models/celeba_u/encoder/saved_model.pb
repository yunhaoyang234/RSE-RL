ж
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
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718ȶ
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
?
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@H* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:@H*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:H*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:H*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:H*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:H*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:H*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?	?*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
?	?*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
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
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?H*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	?H*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:H*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:H*
dtype0

NoOpNoOp
?,
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?,
value?+B?+ B?+
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
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
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
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasz/kernelz/bias*
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
GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_52626
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpz/kernel/Read/ReadVariableOpz/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8? *'
f"R 
__inference__traced_save_53522
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasconv2d_3/kernelconv2d_3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasz/kernelz/biasdense_1/kerneldense_1/bias*
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
GPU2*0J 8? **
f%R#
!__inference__traced_restore_53586??
?
?
(__inference_conv2d_3_layer_call_fn_53144

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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_517482
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_52011

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
?
?
3__inference_batch_normalization_layer_call_fn_53222

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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_520112
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
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_53364

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
?
?
&__inference_conv2d_layer_call_fn_52994

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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_516522
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
<__inference_z_layer_call_and_return_conditional_losses_53344

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
?"
?
A__inference_conv2d_layer_call_and_return_conditional_losses_53020

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp?
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
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_53424Q
7conv2d_2_kernel_regularizer_abs_readvariableop_resource: @
identity??.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_2_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_2/kernel/Regularizer/add_1:z:0/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp
?
?
'__inference_encoder_layer_call_fn_52301
input_1!
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
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_522172
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
_user_specified_name	input_1
?
?
%__inference_dense_layer_call_fn_53314

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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_518002
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
?
?
__inference_loss_fn_0_53384O
5conv2d_kernel_regularizer_abs_readvariableop_resource:
identity??,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp?
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp5conv2d_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp5conv2d_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
IdentityIdentity#conv2d/kernel/Regularizer/add_1:z:0-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp
?#
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_51748

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
̖
?
B__inference_encoder_layer_call_and_return_conditional_losses_51901

inputs&
conv2d_51653:
conv2d_51655:(
conv2d_1_51685: 
conv2d_1_51687: (
conv2d_2_51717: @
conv2d_2_51719:@(
conv2d_3_51749:@H
conv2d_3_51751:H'
batch_normalization_51772:H'
batch_normalization_51774:H'
batch_normalization_51776:H'
batch_normalization_51778:H
dense_51801:
?	?
dense_51803:	? 
dense_1_51818:	?H
dense_1_51820:H
z_51834:	?H
z_51836:H
identity

identity_1??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp? conv2d_1/StatefulPartitionedCall?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_2/StatefulPartitionedCall?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_51653conv2d_51655*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_516522 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_51685conv2d_1_51687*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_516842"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_51717conv2d_2_51719*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_517162"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_51749conv2d_3_51751*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_517482"
 conv2d_3/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_51772batch_normalization_51774batch_normalization_51776batch_normalization_51778*
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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_517712-
+batch_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_517872
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_51801dense_51803*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_518002
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_51818dense_1_51820*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_518172!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_51834z_51836*
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
GPU2*0J 8? *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_518332
z/StatefulPartitionedCall?
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_51653*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_51653*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_51685*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_51685*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_51717*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_51717*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_51749*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_51749*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
B__inference_encoder_layer_call_and_return_conditional_losses_52841

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@H6
(conv2d_3_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
?	?4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?H5
'dense_1_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_3/Relu?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_1/BiasAdd|
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_1/Softplus?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
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
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentityz/BiasAdd:output:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identitydense_1/Softplus:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53294

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
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_53305

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
?O
?
!__inference__traced_restore_53586
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel: .
 assignvariableop_3_conv2d_1_bias: <
"assignvariableop_4_conv2d_2_kernel: @.
 assignvariableop_5_conv2d_2_bias:@<
"assignvariableop_6_conv2d_3_kernel:@H.
 assignvariableop_7_conv2d_3_bias:H:
,assignvariableop_8_batch_normalization_gamma:H9
+assignvariableop_9_batch_normalization_beta:HA
3assignvariableop_10_batch_normalization_moving_mean:HE
7assignvariableop_11_batch_normalization_moving_variance:H4
 assignvariableop_12_dense_kernel:
?	?-
assignvariableop_13_dense_bias:	?/
assignvariableop_14_z_kernel:	?H(
assignvariableop_15_z_bias:H5
"assignvariableop_16_dense_1_kernel:	?H.
 assignvariableop_17_dense_1_bias:H
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
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp3assignvariableop_10_batch_normalization_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp7assignvariableop_11_batch_normalization_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_dense_biasIdentity_13:output:0"/device:CPU:0*
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
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_1_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_1_biasIdentity_17:output:0"/device:CPU:0*
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
ʖ
?
B__inference_encoder_layer_call_and_return_conditional_losses_52217

inputs&
conv2d_52110:
conv2d_52112:(
conv2d_1_52115: 
conv2d_1_52117: (
conv2d_2_52120: @
conv2d_2_52122:@(
conv2d_3_52125:@H
conv2d_3_52127:H'
batch_normalization_52130:H'
batch_normalization_52132:H'
batch_normalization_52134:H'
batch_normalization_52136:H
dense_52140:
?	?
dense_52142:	? 
dense_1_52145:	?H
dense_1_52147:H
z_52150:	?H
z_52152:H
identity

identity_1??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp? conv2d_1/StatefulPartitionedCall?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_2/StatefulPartitionedCall?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_52110conv2d_52112*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_516522 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_52115conv2d_1_52117*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_516842"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_52120conv2d_2_52122*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_517162"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_52125conv2d_3_52127*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_517482"
 conv2d_3/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_52130batch_normalization_52132batch_normalization_52134batch_normalization_52136*
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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_520112-
+batch_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_517872
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52140dense_52142*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_518002
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52145dense_1_52147*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_518172!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_52150z_52152*
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
GPU2*0J 8? *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_518332
z/StatefulPartitionedCall?
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_52110*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_52110*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_52115*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_52115*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_52120*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_52120*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_52125*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_52125*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_53196

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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_515592
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
#__inference_signature_wrapper_52626
input_1!
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
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_514932
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
_user_specified_name	input_1
?#
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_51684

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
C
'__inference_flatten_layer_call_fn_53299

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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_517872
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
?
?
(__inference_conv2d_2_layer_call_fn_53094

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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_517162
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
?

?
@__inference_dense_layer_call_and_return_conditional_losses_51800

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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53240

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
__inference_loss_fn_3_53444Q
7conv2d_3_kernel_regularizer_abs_readvariableop_resource:@H
identity??.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_3_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_3/kernel/Regularizer/add_1:z:0/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp
??
?
B__inference_encoder_layer_call_and_return_conditional_losses_52970

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@A
'conv2d_3_conv2d_readvariableop_resource:@H6
(conv2d_3_biasadd_readvariableop_resource:H9
+batch_normalization_readvariableop_resource:H;
-batch_normalization_readvariableop_1_resource:HJ
<batch_normalization_fusedbatchnormv3_readvariableop_resource:HL
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H8
$dense_matmul_readvariableop_resource:
?	?4
%dense_biasadd_readvariableop_resource:	?9
&dense_1_matmul_readvariableop_resource:	?H5
'dense_1_biasadd_readvariableop_resource:H3
 z_matmul_readvariableop_resource:	?H/
!z_biasadd_readvariableop_resource:H
identity

identity_1??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?z/BiasAdd/ReadVariableOp?z/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_1/Relu?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dconv2d_1/Relu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_2/Relu?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
conv2d_3/Relu?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_3/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
flatten/Const?
flatten/ReshapeReshape(batch_normalization/FusedBatchNormV3:y:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
dense_1/BiasAdd|
dense_1/SoftplusSoftplusdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
dense_1/Softplus?
z/MatMul/ReadVariableOpReadVariableOp z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02
z/MatMul/ReadVariableOp?
z/MatMulMatMuldense/Relu:activations:0z/MatMul/ReadVariableOp:value:0*
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
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?	
IdentityIdentityz/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?	

Identity_1Identitydense_1/Softplus:activations:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^z/BiasAdd/ReadVariableOp^z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp24
z/BiasAdd/ReadVariableOpz/BiasAdd/ReadVariableOp22
z/MatMul/ReadVariableOpz/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53258

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
?

?
@__inference_dense_layer_call_and_return_conditional_losses_53325

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_51716

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
!__inference_z_layer_call_fn_53334

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
GPU2*0J 8? *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_518332
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
?.
?
__inference__traced_save_53522
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop'
#savev2_z_kernel_read_readvariableop%
!savev2_z_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
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
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop#savev2_z_kernel_read_readvariableop!savev2_z_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
͖
?
B__inference_encoder_layer_call_and_return_conditional_losses_52521
input_1&
conv2d_52414:
conv2d_52416:(
conv2d_1_52419: 
conv2d_1_52421: (
conv2d_2_52424: @
conv2d_2_52426:@(
conv2d_3_52429:@H
conv2d_3_52431:H'
batch_normalization_52434:H'
batch_normalization_52436:H'
batch_normalization_52438:H'
batch_normalization_52440:H
dense_52444:
?	?
dense_52446:	? 
dense_1_52449:	?H
dense_1_52451:H
z_52454:	?H
z_52456:H
identity

identity_1??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp? conv2d_1/StatefulPartitionedCall?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_2/StatefulPartitionedCall?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_52414conv2d_52416*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_516522 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_52419conv2d_1_52421*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_516842"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_52424conv2d_2_52426*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_517162"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_52429conv2d_3_52431*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_517482"
 conv2d_3/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_52434batch_normalization_52436batch_normalization_52438batch_normalization_52440*
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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_520112-
+batch_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_517872
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52444dense_52446*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_518002
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52449dense_1_52451*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_518172!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_52454z_52456*
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
GPU2*0J 8? *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_518332
z/StatefulPartitionedCall?
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_52414*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_52414*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_52419*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_52419*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_52424*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_52424*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_52429*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_52429*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
ϖ
?
B__inference_encoder_layer_call_and_return_conditional_losses_52411
input_1&
conv2d_52304:
conv2d_52306:(
conv2d_1_52309: 
conv2d_1_52311: (
conv2d_2_52314: @
conv2d_2_52316:@(
conv2d_3_52319:@H
conv2d_3_52321:H'
batch_normalization_52324:H'
batch_normalization_52326:H'
batch_normalization_52328:H'
batch_normalization_52330:H
dense_52334:
?	?
dense_52336:	? 
dense_1_52339:	?H
dense_1_52341:H
z_52344:	?H
z_52346:H
identity

identity_1??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp? conv2d_1/StatefulPartitionedCall?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp? conv2d_2/StatefulPartitionedCall?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp? conv2d_3/StatefulPartitionedCall?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?z/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_52304conv2d_52306*
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
GPU2*0J 8? *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_516522 
conv2d/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_52309conv2d_1_52311*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_516842"
 conv2d_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0conv2d_2_52314conv2d_2_52316*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_517162"
 conv2d_2/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_52319conv2d_3_52321*
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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_517482"
 conv2d_3/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_52324batch_normalization_52326batch_normalization_52328batch_normalization_52330*
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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_517712-
+batch_normalization/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
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
GPU2*0J 8? *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_517872
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_52334dense_52336*
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
GPU2*0J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_518002
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_52339dense_1_52341*
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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_518172!
dense_1/StatefulPartitionedCall?
z/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0z_52344z_52346*
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
GPU2*0J 8? *E
f@R>
<__inference_z_layer_call_and_return_conditional_losses_518332
z/StatefulPartitionedCall?
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_52304*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_52304*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_1_52309*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_1_52309*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_2_52314*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_2_52314*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_3_52319*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_3_52319*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentity"z/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identity(dense_1/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp!^conv2d_1/StatefulPartitionedCall/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp!^conv2d_2/StatefulPartitionedCall/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp!^conv2d_3/StatefulPartitionedCall/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^z/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2`
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2`
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall26
z/StatefulPartitionedCallz/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
'__inference_dense_1_layer_call_fn_53353

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
GPU2*0J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_518172
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
?

?
B__inference_dense_1_layer_call_and_return_conditional_losses_51817

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
?
?
__inference_loss_fn_1_53404Q
7conv2d_1_kernel_regularizer_abs_readvariableop_resource: 
identity??.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp7conv2d_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOp7conv2d_1_kernel_regularizer_abs_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
IdentityIdentity%conv2d_1/kernel/Regularizer/add_1:z:0/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2`
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp
?#
?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53170

inputs8
conv2d_readvariableop_resource:@H-
biasadd_readvariableop_resource:H
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_3/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_3/kernel/Regularizer/Const?
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype020
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_3/kernel/Regularizer/AbsAbs6conv2d_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2!
conv2d_3/kernel/Regularizer/Abs?
#conv2d_3/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_1?
conv2d_3/kernel/Regularizer/SumSum#conv2d_3/kernel/Regularizer/Abs:y:0,conv2d_3/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/Sum?
!conv2d_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_3/kernel/Regularizer/mul/x?
conv2d_3/kernel/Regularizer/mulMul*conv2d_3/kernel/Regularizer/mul/x:output:0(conv2d_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/mul?
conv2d_3/kernel/Regularizer/addAddV2*conv2d_3/kernel/Regularizer/Const:output:0#conv2d_3/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_3/kernel/Regularizer/add?
1conv2d_3/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype023
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_3/kernel/Regularizer/SquareSquare9conv2d_3/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@H2$
"conv2d_3/kernel/Regularizer/Square?
#conv2d_3/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_3/kernel/Regularizer/Const_2?
!conv2d_3/kernel/Regularizer/Sum_1Sum&conv2d_3/kernel/Regularizer/Square:y:0,conv2d_3/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/Sum_1?
#conv2d_3/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_3/kernel/Regularizer/mul_1/x?
!conv2d_3/kernel/Regularizer/mul_1Mul,conv2d_3/kernel/Regularizer/mul_1/x:output:0*conv2d_3/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/mul_1?
!conv2d_3/kernel/Regularizer/add_1AddV2#conv2d_3/kernel/Regularizer/add:z:0%conv2d_3/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_3/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_3/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp.conv2d_3/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_3/kernel/Regularizer/Square/ReadVariableOp1conv2d_3/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
(__inference_conv2d_1_layer_call_fn_53044

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
GPU2*0J 8? *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_516842
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
?l
?
 __inference__wrapped_model_51493
input_1G
-encoder_conv2d_conv2d_readvariableop_resource:<
.encoder_conv2d_biasadd_readvariableop_resource:I
/encoder_conv2d_1_conv2d_readvariableop_resource: >
0encoder_conv2d_1_biasadd_readvariableop_resource: I
/encoder_conv2d_2_conv2d_readvariableop_resource: @>
0encoder_conv2d_2_biasadd_readvariableop_resource:@I
/encoder_conv2d_3_conv2d_readvariableop_resource:@H>
0encoder_conv2d_3_biasadd_readvariableop_resource:HA
3encoder_batch_normalization_readvariableop_resource:HC
5encoder_batch_normalization_readvariableop_1_resource:HR
Dencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource:HT
Fencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:H@
,encoder_dense_matmul_readvariableop_resource:
?	?<
-encoder_dense_biasadd_readvariableop_resource:	?A
.encoder_dense_1_matmul_readvariableop_resource:	?H=
/encoder_dense_1_biasadd_readvariableop_resource:H;
(encoder_z_matmul_readvariableop_resource:	?H7
)encoder_z_biasadd_readvariableop_resource:H
identity

identity_1??;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp?=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?*encoder/batch_normalization/ReadVariableOp?,encoder/batch_normalization/ReadVariableOp_1?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?'encoder/conv2d_2/BiasAdd/ReadVariableOp?&encoder/conv2d_2/Conv2D/ReadVariableOp?'encoder/conv2d_3/BiasAdd/ReadVariableOp?&encoder/conv2d_3/Conv2D/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?&encoder/dense_1/BiasAdd/ReadVariableOp?%encoder/dense_1/MatMul/ReadVariableOp? encoder/z/BiasAdd/ReadVariableOp?encoder/z/MatMul/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2Dinput_1,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d/BiasAdd?
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
encoder/conv2d/Relu?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOp?
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
encoder/conv2d_1/Conv2D?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOp?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_1/BiasAdd?
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_1/Relu?
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOp?
encoder/conv2d_2/Conv2DConv2D#encoder/conv2d_1/Relu:activations:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
encoder/conv2d_2/Conv2D?
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOp?
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_2/BiasAdd?
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_2/Relu?
&encoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@H*
dtype02(
&encoder/conv2d_3/Conv2D/ReadVariableOp?
encoder/conv2d_3/Conv2DConv2D#encoder/conv2d_2/Relu:activations:0.encoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H*
paddingSAME*
strides
2
encoder/conv2d_3/Conv2D?
'encoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02)
'encoder/conv2d_3/BiasAdd/ReadVariableOp?
encoder/conv2d_3/BiasAddBiasAdd encoder/conv2d_3/Conv2D:output:0/encoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????H2
encoder/conv2d_3/BiasAdd?
encoder/conv2d_3/ReluRelu!encoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????H2
encoder/conv2d_3/Relu?
*encoder/batch_normalization/ReadVariableOpReadVariableOp3encoder_batch_normalization_readvariableop_resource*
_output_shapes
:H*
dtype02,
*encoder/batch_normalization/ReadVariableOp?
,encoder/batch_normalization/ReadVariableOp_1ReadVariableOp5encoder_batch_normalization_readvariableop_1_resource*
_output_shapes
:H*
dtype02.
,encoder/batch_normalization/ReadVariableOp_1?
;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:H*
dtype02=
;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp?
=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:H*
dtype02?
=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
,encoder/batch_normalization/FusedBatchNormV3FusedBatchNormV3#encoder/conv2d_3/Relu:activations:02encoder/batch_normalization/ReadVariableOp:value:04encoder/batch_normalization/ReadVariableOp_1:value:0Cencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Eencoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????H:H:H:H:H:*
epsilon%o?:*
is_training( 2.
,encoder/batch_normalization/FusedBatchNormV3
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape0encoder/batch_normalization/FusedBatchNormV3:y:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????	2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
?	?*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
encoder/dense/BiasAdd?
encoder/dense/ReluReluencoder/dense/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
encoder/dense/Relu?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMul encoder/dense/Relu:activations:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:H*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_1/BiasAdd?
encoder/dense_1/SoftplusSoftplus encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????H2
encoder/dense_1/Softplus?
encoder/z/MatMul/ReadVariableOpReadVariableOp(encoder_z_matmul_readvariableop_resource*
_output_shapes
:	?H*
dtype02!
encoder/z/MatMul/ReadVariableOp?
encoder/z/MatMulMatMul encoder/dense/Relu:activations:0'encoder/z/MatMul/ReadVariableOp:value:0*
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
encoder/z/BiasAdd?
IdentityIdentity&encoder/dense_1/Softplus:activations:0<^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp>^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^encoder/batch_normalization/ReadVariableOp-^encoder/batch_normalization/ReadVariableOp_1&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity?

Identity_1Identityencoder/z/BiasAdd:output:0<^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp>^encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^encoder/batch_normalization/ReadVariableOp-^encoder/batch_normalization/ReadVariableOp_1&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp(^encoder/conv2d_3/BiasAdd/ReadVariableOp'^encoder/conv2d_3/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp!^encoder/z/BiasAdd/ReadVariableOp ^encoder/z/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????H2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:?????????: : : : : : : : : : : : : : : : : : 2z
;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp;encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp2~
=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=encoder/batch_normalization/FusedBatchNormV3/ReadVariableOp_12X
*encoder/batch_normalization/ReadVariableOp*encoder/batch_normalization/ReadVariableOp2\
,encoder/batch_normalization/ReadVariableOp_1,encoder/batch_normalization/ReadVariableOp_12N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2R
'encoder/conv2d_3/BiasAdd/ReadVariableOp'encoder/conv2d_3/BiasAdd/ReadVariableOp2P
&encoder/conv2d_3/Conv2D/ReadVariableOp&encoder/conv2d_3/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp2D
 encoder/z/BiasAdd/ReadVariableOp encoder/z/BiasAdd/ReadVariableOp2B
encoder/z/MatMul/ReadVariableOpencoder/z/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_51559

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
?#
?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53070

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_1/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_1/kernel/Regularizer/Const?
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype020
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_1/kernel/Regularizer/AbsAbs6conv2d_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Abs?
#conv2d_1/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_1?
conv2d_1/kernel/Regularizer/SumSum#conv2d_1/kernel/Regularizer/Abs:y:0,conv2d_1/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/Sum?
!conv2d_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_1/kernel/Regularizer/mul/x?
conv2d_1/kernel/Regularizer/mulMul*conv2d_1/kernel/Regularizer/mul/x:output:0(conv2d_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/mul?
conv2d_1/kernel/Regularizer/addAddV2*conv2d_1/kernel/Regularizer/Const:output:0#conv2d_1/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_1/kernel/Regularizer/add?
1conv2d_1/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype023
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_1/kernel/Regularizer/SquareSquare9conv2d_1/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: 2$
"conv2d_1/kernel/Regularizer/Square?
#conv2d_1/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_1/kernel/Regularizer/Const_2?
!conv2d_1/kernel/Regularizer/Sum_1Sum&conv2d_1/kernel/Regularizer/Square:y:0,conv2d_1/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/Sum_1?
#conv2d_1/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_1/kernel/Regularizer/mul_1/x?
!conv2d_1/kernel/Regularizer/mul_1Mul,conv2d_1/kernel/Regularizer/mul_1/x:output:0*conv2d_1/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/mul_1?
!conv2d_1/kernel/Regularizer/add_1AddV2#conv2d_1/kernel/Regularizer/add:z:0%conv2d_1/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_1/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_1/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp.conv2d_1/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_1/kernel/Regularizer/Square/ReadVariableOp1conv2d_1/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_53183

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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_515152
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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53276

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
?"
?
A__inference_conv2d_layer_call_and_return_conditional_losses_51652

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?,conv2d/kernel/Regularizer/Abs/ReadVariableOp?/conv2d/kernel/Regularizer/Square/ReadVariableOp?
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
conv2d/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
conv2d/kernel/Regularizer/Const?
,conv2d/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,conv2d/kernel/Regularizer/Abs/ReadVariableOp?
conv2d/kernel/Regularizer/AbsAbs4conv2d/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:2
conv2d/kernel/Regularizer/Abs?
!conv2d/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_1?
conv2d/kernel/Regularizer/SumSum!conv2d/kernel/Regularizer/Abs:y:0*conv2d/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/Sum?
conv2d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2!
conv2d/kernel/Regularizer/mul/x?
conv2d/kernel/Regularizer/mulMul(conv2d/kernel/Regularizer/mul/x:output:0&conv2d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/mul?
conv2d/kernel/Regularizer/addAddV2(conv2d/kernel/Regularizer/Const:output:0!conv2d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2
conv2d/kernel/Regularizer/add?
/conv2d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype021
/conv2d/kernel/Regularizer/Square/ReadVariableOp?
 conv2d/kernel/Regularizer/SquareSquare7conv2d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:2"
 conv2d/kernel/Regularizer/Square?
!conv2d/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2#
!conv2d/kernel/Regularizer/Const_2?
conv2d/kernel/Regularizer/Sum_1Sum$conv2d/kernel/Regularizer/Square:y:0*conv2d/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/Sum_1?
!conv2d/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d/kernel/Regularizer/mul_1/x?
conv2d/kernel/Regularizer/mul_1Mul*conv2d/kernel/Regularizer/mul_1/x:output:0(conv2d/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/mul_1?
conv2d/kernel/Regularizer/add_1AddV2!conv2d/kernel/Regularizer/add:z:0#conv2d/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
conv2d/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp-^conv2d/kernel/Regularizer/Abs/ReadVariableOp0^conv2d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2\
,conv2d/kernel/Regularizer/Abs/ReadVariableOp,conv2d/kernel/Regularizer/Abs/ReadVariableOp2b
/conv2d/kernel/Regularizer/Square/ReadVariableOp/conv2d/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_encoder_layer_call_fn_51942
input_1!
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
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_519012
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
_user_specified_name	input_1
?
?
'__inference_encoder_layer_call_fn_52669

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
GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_519012
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
?
?
'__inference_encoder_layer_call_fn_52712

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
GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_522172
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
?
^
B__inference_flatten_layer_call_and_return_conditional_losses_51787

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
?#
?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53120

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
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
!conv2d_2/kernel/Regularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!conv2d_2/kernel/Regularizer/Const?
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype020
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp?
conv2d_2/kernel/Regularizer/AbsAbs6conv2d_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2!
conv2d_2/kernel/Regularizer/Abs?
#conv2d_2/kernel/Regularizer/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_1?
conv2d_2/kernel/Regularizer/SumSum#conv2d_2/kernel/Regularizer/Abs:y:0,conv2d_2/kernel/Regularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/Sum?
!conv2d_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2#
!conv2d_2/kernel/Regularizer/mul/x?
conv2d_2/kernel/Regularizer/mulMul*conv2d_2/kernel/Regularizer/mul/x:output:0(conv2d_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/mul?
conv2d_2/kernel/Regularizer/addAddV2*conv2d_2/kernel/Regularizer/Const:output:0#conv2d_2/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
conv2d_2/kernel/Regularizer/add?
1conv2d_2/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype023
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp?
"conv2d_2/kernel/Regularizer/SquareSquare9conv2d_2/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
: @2$
"conv2d_2/kernel/Regularizer/Square?
#conv2d_2/kernel/Regularizer/Const_2Const*
_output_shapes
:*
dtype0*%
valueB"             2%
#conv2d_2/kernel/Regularizer/Const_2?
!conv2d_2/kernel/Regularizer/Sum_1Sum&conv2d_2/kernel/Regularizer/Square:y:0,conv2d_2/kernel/Regularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/Sum_1?
#conv2d_2/kernel/Regularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *
?#<2%
#conv2d_2/kernel/Regularizer/mul_1/x?
!conv2d_2/kernel/Regularizer/mul_1Mul,conv2d_2/kernel/Regularizer/mul_1/x:output:0*conv2d_2/kernel/Regularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/mul_1?
!conv2d_2/kernel/Regularizer/add_1AddV2#conv2d_2/kernel/Regularizer/add:z:0%conv2d_2/kernel/Regularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!conv2d_2/kernel/Regularizer/add_1?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp/^conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2^conv2d_2/kernel/Regularizer/Square/ReadVariableOp*
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
.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp.conv2d_2/kernel/Regularizer/Abs/ReadVariableOp2f
1conv2d_2/kernel/Regularizer/Square/ReadVariableOp1conv2d_2/kernel/Regularizer/Square/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_51771

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
?
?
3__inference_batch_normalization_layer_call_fn_53209

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
GPU2*0J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_517712
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
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_51515

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
?	
?
<__inference_z_layer_call_and_return_conditional_losses_51833

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
input_18
serving_default_input_1:0?????????;
dense_10
StatefulPartitionedCall:0?????????H5
z0
StatefulPartitionedCall:1?????????Htensorflow/serving/predict:??
?m
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
*{&call_and_return_all_conditional_losses"?i
_tf_keras_network?i{"name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z", 0, 0], ["dense_1", 0, 0]]}, "shared_object_id": 29, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 16, 16, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["conv2d", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "z", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 28}], "input_layers": [["input_1", 0, 0]], "output_layers": [["z", 0, 0], ["dense_1", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
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
{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 1]}}
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
{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 32}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
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
{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 33}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 32]}}
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
{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 72, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.009999999776482582, "l2": 0.009999999776482582}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 64]}}
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
_tf_keras_layer?	{"name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "gamma_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 15}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "moving_variance_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 72}}, "shared_object_id": 35}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 72]}}
?
1regularization_losses
2trainable_variables
3	variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["batch_normalization", 0, 0, {}]]], "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 36}}
?

5kernel
6bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatten", 0, 0, {}]]], "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
?

;kernel
<bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "z", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "z", "trainable": true, "dtype": "float32", "units": 72, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?	

Akernel
Bbias
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 72, "activation": "softplus", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
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
':%2conv2d/kernel
:2conv2d/bias
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
):' 2conv2d_1/kernel
: 2conv2d_1/bias
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
):' @2conv2d_2/kernel
:@2conv2d_2/bias
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
):'@H2conv2d_3/kernel
:H2conv2d_3/bias
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
':%H2batch_normalization/gamma
&:$H2batch_normalization/beta
/:-H (2batch_normalization/moving_mean
3:1H (2#batch_normalization/moving_variance
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
 :
?	?2dense/kernel
:?2
dense/bias
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
!:	?H2dense_1/kernel
:H2dense_1/bias
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
 __inference__wrapped_model_51493?
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
input_1?????????
?2?
'__inference_encoder_layer_call_fn_51942
'__inference_encoder_layer_call_fn_52669
'__inference_encoder_layer_call_fn_52712
'__inference_encoder_layer_call_fn_52301?
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
B__inference_encoder_layer_call_and_return_conditional_losses_52841
B__inference_encoder_layer_call_and_return_conditional_losses_52970
B__inference_encoder_layer_call_and_return_conditional_losses_52411
B__inference_encoder_layer_call_and_return_conditional_losses_52521?
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
&__inference_conv2d_layer_call_fn_52994?
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
A__inference_conv2d_layer_call_and_return_conditional_losses_53020?
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
(__inference_conv2d_1_layer_call_fn_53044?
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53070?
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
(__inference_conv2d_2_layer_call_fn_53094?
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53120?
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
(__inference_conv2d_3_layer_call_fn_53144?
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53170?
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
3__inference_batch_normalization_layer_call_fn_53183
3__inference_batch_normalization_layer_call_fn_53196
3__inference_batch_normalization_layer_call_fn_53209
3__inference_batch_normalization_layer_call_fn_53222?
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
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53240
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53258
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53276
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53294?
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
'__inference_flatten_layer_call_fn_53299?
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
B__inference_flatten_layer_call_and_return_conditional_losses_53305?
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
%__inference_dense_layer_call_fn_53314?
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
@__inference_dense_layer_call_and_return_conditional_losses_53325?
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
!__inference_z_layer_call_fn_53334?
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
<__inference_z_layer_call_and_return_conditional_losses_53344?
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
'__inference_dense_1_layer_call_fn_53353?
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
B__inference_dense_1_layer_call_and_return_conditional_losses_53364?
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
__inference_loss_fn_0_53384?
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
__inference_loss_fn_1_53404?
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
__inference_loss_fn_2_53424?
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
__inference_loss_fn_3_53444?
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
#__inference_signature_wrapper_52626input_1"?
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
 __inference__wrapped_model_51493?"#)*+,56AB;<8?5
.?+
)?&
input_1?????????
? "S?P
,
dense_1!?
dense_1?????????H
 
z?
z?????????H?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53240?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p 
? "??<
5?2
0+???????????????????????????H
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53258?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p
? "??<
5?2
0+???????????????????????????H
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53276r)*+,;?8
1?.
(?%
inputs?????????H
p 
? "-?*
#? 
0?????????H
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_53294r)*+,;?8
1?.
(?%
inputs?????????H
p
? "-?*
#? 
0?????????H
? ?
3__inference_batch_normalization_layer_call_fn_53183?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p 
? "2?/+???????????????????????????H?
3__inference_batch_normalization_layer_call_fn_53196?)*+,M?J
C?@
:?7
inputs+???????????????????????????H
p
? "2?/+???????????????????????????H?
3__inference_batch_normalization_layer_call_fn_53209e)*+,;?8
1?.
(?%
inputs?????????H
p 
? " ??????????H?
3__inference_batch_normalization_layer_call_fn_53222e)*+,;?8
1?.
(?%
inputs?????????H
p
? " ??????????H?
C__inference_conv2d_1_layer_call_and_return_conditional_losses_53070l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_1_layer_call_fn_53044_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_2_layer_call_and_return_conditional_losses_53120l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
(__inference_conv2d_2_layer_call_fn_53094_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
C__inference_conv2d_3_layer_call_and_return_conditional_losses_53170l"#7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????H
? ?
(__inference_conv2d_3_layer_call_fn_53144_"#7?4
-?*
(?%
inputs?????????@
? " ??????????H?
A__inference_conv2d_layer_call_and_return_conditional_losses_53020l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_conv2d_layer_call_fn_52994_7?4
-?*
(?%
inputs?????????
? " ???????????
B__inference_dense_1_layer_call_and_return_conditional_losses_53364]AB0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? {
'__inference_dense_1_layer_call_fn_53353PAB0?-
&?#
!?
inputs??????????
? "??????????H?
@__inference_dense_layer_call_and_return_conditional_losses_53325^560?-
&?#
!?
inputs??????????	
? "&?#
?
0??????????
? z
%__inference_dense_layer_call_fn_53314Q560?-
&?#
!?
inputs??????????	
? "????????????
B__inference_encoder_layer_call_and_return_conditional_losses_52411?"#)*+,56AB;<@?=
6?3
)?&
input_1?????????
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
B__inference_encoder_layer_call_and_return_conditional_losses_52521?"#)*+,56AB;<@?=
6?3
)?&
input_1?????????
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
B__inference_encoder_layer_call_and_return_conditional_losses_52841?"#)*+,56AB;<??<
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
B__inference_encoder_layer_call_and_return_conditional_losses_52970?"#)*+,56AB;<??<
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
'__inference_encoder_layer_call_fn_51942?"#)*+,56AB;<@?=
6?3
)?&
input_1?????????
p 

 
? "=?:
?
0?????????H
?
1?????????H?
'__inference_encoder_layer_call_fn_52301?"#)*+,56AB;<@?=
6?3
)?&
input_1?????????
p

 
? "=?:
?
0?????????H
?
1?????????H?
'__inference_encoder_layer_call_fn_52669?"#)*+,56AB;<??<
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
'__inference_encoder_layer_call_fn_52712?"#)*+,56AB;<??<
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
B__inference_flatten_layer_call_and_return_conditional_losses_53305a7?4
-?*
(?%
inputs?????????H
? "&?#
?
0??????????	
? 
'__inference_flatten_layer_call_fn_53299T7?4
-?*
(?%
inputs?????????H
? "???????????	:
__inference_loss_fn_0_53384?

? 
? "? :
__inference_loss_fn_1_53404?

? 
? "? :
__inference_loss_fn_2_53424?

? 
? "? :
__inference_loss_fn_3_53444"?

? 
? "? ?
#__inference_signature_wrapper_52626?"#)*+,56AB;<C?@
? 
9?6
4
input_1)?&
input_1?????????"S?P
,
dense_1!?
dense_1?????????H
 
z?
z?????????H?
<__inference_z_layer_call_and_return_conditional_losses_53344];<0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????H
? u
!__inference_z_layer_call_fn_53334P;<0?-
&?#
!?
inputs??????????
? "??????????H