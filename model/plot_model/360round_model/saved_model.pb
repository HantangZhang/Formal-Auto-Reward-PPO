��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
t
Adam/pi/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/pi/bias/v
m
"Adam/pi/bias/v/Read/ReadVariableOpReadVariableOpAdam/pi/bias/v*
_output_shapes
:*
dtype0
|
Adam/pi/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/pi/kernel/v
u
$Adam/pi/kernel/v/Read/ReadVariableOpReadVariableOpAdam/pi/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/mlp_fc3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc3/bias/v
w
'Adam/mlp_fc3/bias/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc3/bias/v*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc3/kernel/v

)Adam/mlp_fc3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc3/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc2/bias/v
w
'Adam/mlp_fc2/bias/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc2/bias/v*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc2/kernel/v

)Adam/mlp_fc2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc2/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc1/bias/v
w
'Adam/mlp_fc1/bias/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc1/bias/v*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc1/kernel/v

)Adam/mlp_fc1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc1/kernel/v*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc0/bias/v
w
'Adam/mlp_fc0/bias/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc0/bias/v*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/mlp_fc0/kernel/v

)Adam/mlp_fc0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/mlp_fc0/kernel/v*
_output_shapes

:@*
dtype0
t
Adam/vf/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/vf/bias/v
m
"Adam/vf/bias/v/Read/ReadVariableOpReadVariableOpAdam/vf/bias/v*
_output_shapes
:*
dtype0
|
Adam/vf/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/vf/kernel/v
u
$Adam/vf/kernel/v/Read/ReadVariableOpReadVariableOpAdam/vf/kernel/v*
_output_shapes

:@*
dtype0
t
Adam/pi/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/pi/bias/m
m
"Adam/pi/bias/m/Read/ReadVariableOpReadVariableOpAdam/pi/bias/m*
_output_shapes
:*
dtype0
|
Adam/pi/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/pi/kernel/m
u
$Adam/pi/kernel/m/Read/ReadVariableOpReadVariableOpAdam/pi/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/mlp_fc3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc3/bias/m
w
'Adam/mlp_fc3/bias/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc3/bias/m*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc3/kernel/m

)Adam/mlp_fc3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc3/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc2/bias/m
w
'Adam/mlp_fc2/bias/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc2/bias/m*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc2/kernel/m

)Adam/mlp_fc2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc2/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc1/bias/m
w
'Adam/mlp_fc1/bias/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc1/bias/m*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*&
shared_nameAdam/mlp_fc1/kernel/m

)Adam/mlp_fc1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc1/kernel/m*
_output_shapes

:@@*
dtype0
~
Adam/mlp_fc0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/mlp_fc0/bias/m
w
'Adam/mlp_fc0/bias/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc0/bias/m*
_output_shapes
:@*
dtype0
�
Adam/mlp_fc0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/mlp_fc0/kernel/m

)Adam/mlp_fc0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/mlp_fc0/kernel/m*
_output_shapes

:@*
dtype0
t
Adam/vf/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/vf/bias/m
m
"Adam/vf/bias/m/Read/ReadVariableOpReadVariableOpAdam/vf/bias/m*
_output_shapes
:*
dtype0
|
Adam/vf/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_nameAdam/vf/kernel/m
u
$Adam/vf/kernel/m/Read/ReadVariableOpReadVariableOpAdam/vf/kernel/m*
_output_shapes

:@*
dtype0
f
pi/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	pi/bias
_
pi/bias/Read/ReadVariableOpReadVariableOppi/bias*
_output_shapes
:*
dtype0
n
	pi/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name	pi/kernel
g
pi/kernel/Read/ReadVariableOpReadVariableOp	pi/kernel*
_output_shapes

:@*
dtype0
p
mlp_fc3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemlp_fc3/bias
i
 mlp_fc3/bias/Read/ReadVariableOpReadVariableOpmlp_fc3/bias*
_output_shapes
:@*
dtype0
x
mlp_fc3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namemlp_fc3/kernel
q
"mlp_fc3/kernel/Read/ReadVariableOpReadVariableOpmlp_fc3/kernel*
_output_shapes

:@@*
dtype0
p
mlp_fc2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemlp_fc2/bias
i
 mlp_fc2/bias/Read/ReadVariableOpReadVariableOpmlp_fc2/bias*
_output_shapes
:@*
dtype0
x
mlp_fc2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namemlp_fc2/kernel
q
"mlp_fc2/kernel/Read/ReadVariableOpReadVariableOpmlp_fc2/kernel*
_output_shapes

:@@*
dtype0
p
mlp_fc1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemlp_fc1/bias
i
 mlp_fc1/bias/Read/ReadVariableOpReadVariableOpmlp_fc1/bias*
_output_shapes
:@*
dtype0
x
mlp_fc1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*
shared_namemlp_fc1/kernel
q
"mlp_fc1/kernel/Read/ReadVariableOpReadVariableOpmlp_fc1/kernel*
_output_shapes

:@@*
dtype0
p
mlp_fc0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namemlp_fc0/bias
i
 mlp_fc0/bias/Read/ReadVariableOpReadVariableOpmlp_fc0/bias*
_output_shapes
:@*
dtype0
x
mlp_fc0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namemlp_fc0/kernel
q
"mlp_fc0/kernel/Read/ReadVariableOpReadVariableOpmlp_fc0/kernel*
_output_shapes

:@*
dtype0
f
vf/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	vf/bias
_
vf/bias/Read/ReadVariableOpReadVariableOpvf/bias*
_output_shapes
:*
dtype0
n
	vf/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_name	vf/kernel
g
vf/kernel/Read/ReadVariableOpReadVariableOp	vf/kernel*
_output_shapes

:@*
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

NoOpNoOp
�P
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�O
value�OB�O B�O
g
train_model
	optimizer

loss_names
get_grad
step
	value

signatures*
Z
policy_network
value_network

	pdtype

value_fc
step
	value*
�
iter

beta_1

beta_2
	decay
learning_rate'm�(m�/m�0m�7m�8m�?m�@m�Gm�Hm�\m�]m�'v�(v�/v�0v�7v�8v�?v�@v�Gv�Hv�\v�]v�*
* 

trace_0* 

trace_0* 

trace_0
trace_1* 
* 
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature*

 matching_fc*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias*
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias*
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias*
<
/0
01
72
83
?4
@5
G6
H7*
<
/0
01
72
83
?4
@5
G6
H7*
* 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_3* 
6
Rtrace_0
Strace_1
Ttrace_2
Utrace_3* 
* 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias*

'0
(1*

'0
(1*
* 
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
YS
VARIABLE_VALUE	vf/kernel6train_model/value_fc/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvf/bias4train_model/value_fc/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

htrace_0* 

itrace_0* 
ys
VARIABLE_VALUEmlp_fc0/kernelQtrain_model/policy_network/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEmlp_fc0/biasOtrain_model/policy_network/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*

otrace_0* 

ptrace_0* 
ys
VARIABLE_VALUEmlp_fc1/kernelQtrain_model/policy_network/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEmlp_fc1/biasOtrain_model/policy_network/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
@1*

?0
@1*
* 
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

vtrace_0* 

wtrace_0* 
ys
VARIABLE_VALUEmlp_fc2/kernelQtrain_model/policy_network/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEmlp_fc2/biasOtrain_model/policy_network/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

G0
H1*

G0
H1*
* 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses*

}trace_0* 

~trace_0* 
ys
VARIABLE_VALUEmlp_fc3/kernelQtrain_model/policy_network/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUEmlp_fc3/biasOtrain_model/policy_network/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1*

\0
]1*
* 
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses*
* 
* 
c]
VARIABLE_VALUE	pi/kernel@train_model/pdtype/matching_fc/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEpi/bias>train_model/pdtype/matching_fc/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
|v
VARIABLE_VALUEAdam/vf/kernel/mRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/vf/bias/mPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc0/kernel/mmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc0/bias/mktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc1/kernel/mmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc1/bias/mktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc2/kernel/mmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc2/bias/mktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc3/kernel/mmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc3/bias/mktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/pi/kernel/m\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/pi/bias/mZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/vf/kernel/vRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/vf/bias/vPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc0/kernel/vmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc0/bias/vktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc1/kernel/vmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc1/bias/vktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc2/kernel/vmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc2/bias/vktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc3/kernel/vmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/mlp_fc3/bias/vktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/pi/kernel/v\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�|
VARIABLE_VALUEAdam/pi/bias/vZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpvf/kernel/Read/ReadVariableOpvf/bias/Read/ReadVariableOp"mlp_fc0/kernel/Read/ReadVariableOp mlp_fc0/bias/Read/ReadVariableOp"mlp_fc1/kernel/Read/ReadVariableOp mlp_fc1/bias/Read/ReadVariableOp"mlp_fc2/kernel/Read/ReadVariableOp mlp_fc2/bias/Read/ReadVariableOp"mlp_fc3/kernel/Read/ReadVariableOp mlp_fc3/bias/Read/ReadVariableOppi/kernel/Read/ReadVariableOppi/bias/Read/ReadVariableOp$Adam/vf/kernel/m/Read/ReadVariableOp"Adam/vf/bias/m/Read/ReadVariableOp)Adam/mlp_fc0/kernel/m/Read/ReadVariableOp'Adam/mlp_fc0/bias/m/Read/ReadVariableOp)Adam/mlp_fc1/kernel/m/Read/ReadVariableOp'Adam/mlp_fc1/bias/m/Read/ReadVariableOp)Adam/mlp_fc2/kernel/m/Read/ReadVariableOp'Adam/mlp_fc2/bias/m/Read/ReadVariableOp)Adam/mlp_fc3/kernel/m/Read/ReadVariableOp'Adam/mlp_fc3/bias/m/Read/ReadVariableOp$Adam/pi/kernel/m/Read/ReadVariableOp"Adam/pi/bias/m/Read/ReadVariableOp$Adam/vf/kernel/v/Read/ReadVariableOp"Adam/vf/bias/v/Read/ReadVariableOp)Adam/mlp_fc0/kernel/v/Read/ReadVariableOp'Adam/mlp_fc0/bias/v/Read/ReadVariableOp)Adam/mlp_fc1/kernel/v/Read/ReadVariableOp'Adam/mlp_fc1/bias/v/Read/ReadVariableOp)Adam/mlp_fc2/kernel/v/Read/ReadVariableOp'Adam/mlp_fc2/bias/v/Read/ReadVariableOp)Adam/mlp_fc3/kernel/v/Read/ReadVariableOp'Adam/mlp_fc3/bias/v/Read/ReadVariableOp$Adam/pi/kernel/v/Read/ReadVariableOp"Adam/pi/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
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
GPU2*0J 8� *)
f$R"
 __inference__traced_save_1861967
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate	vf/kernelvf/biasmlp_fc0/kernelmlp_fc0/biasmlp_fc1/kernelmlp_fc1/biasmlp_fc2/kernelmlp_fc2/biasmlp_fc3/kernelmlp_fc3/bias	pi/kernelpi/biasAdam/vf/kernel/mAdam/vf/bias/mAdam/mlp_fc0/kernel/mAdam/mlp_fc0/bias/mAdam/mlp_fc1/kernel/mAdam/mlp_fc1/bias/mAdam/mlp_fc2/kernel/mAdam/mlp_fc2/bias/mAdam/mlp_fc3/kernel/mAdam/mlp_fc3/bias/mAdam/pi/kernel/mAdam/pi/bias/mAdam/vf/kernel/vAdam/vf/bias/vAdam/mlp_fc0/kernel/vAdam/mlp_fc0/bias/vAdam/mlp_fc1/kernel/vAdam/mlp_fc1/bias/vAdam/mlp_fc2/kernel/vAdam/mlp_fc2/bias/vAdam/mlp_fc3/kernel/vAdam/mlp_fc3/bias/vAdam/pi/kernel/vAdam/pi/bias/v*5
Tin.
,2**
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
GPU2*0J 8� *,
f'R%
#__inference__traced_restore_1862100�
�
�
)__inference_mlp_fc2_layer_call_fn_1861792

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�9
�
(__inference___backward_value_52003_52049
placeholderF
Bgradients_dense_1_matmul_grad_matmul_dense_1_matmul_readvariableop=
9gradients_dense_1_matmul_grad_matmul_1_model_mlp_fc3_tanhR
Ngradients_model_mlp_fc3_matmul_grad_matmul_model_mlp_fc3_matmul_readvariableopC
?gradients_model_mlp_fc3_matmul_grad_matmul_1_model_mlp_fc2_tanhR
Ngradients_model_mlp_fc2_matmul_grad_matmul_model_mlp_fc2_matmul_readvariableopC
?gradients_model_mlp_fc2_matmul_grad_matmul_1_model_mlp_fc1_tanhR
Ngradients_model_mlp_fc1_matmul_grad_matmul_model_mlp_fc1_matmul_readvariableopC
?gradients_model_mlp_fc1_matmul_grad_matmul_1_model_mlp_fc0_tanhR
Ngradients_model_mlp_fc0_matmul_grad_matmul_model_mlp_fc0_matmul_readvariableop;
7gradients_model_mlp_fc0_matmul_grad_matmul_1_model_cast
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9R
gradients/grad_ys_0Identityplaceholder*
T0*
_output_shapes	
:�m
gradients/Squeeze_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradients/Squeeze_grad/ReshapeReshapegradients/grad_ys_0:output:0%gradients/Squeeze_grad/Shape:output:0*
T0*
_output_shapes
:	��
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/Squeeze_grad/Reshape:output:0*
T0*
_output_shapes
:�
$gradients/dense_1/MatMul_grad/MatMulMatMul'gradients/Squeeze_grad/Reshape:output:0Bgradients_dense_1_matmul_grad_matmul_dense_1_matmul_readvariableop*
T0*
_output_shapes
:	�@*
transpose_b(�
&gradients/dense_1/MatMul_grad/MatMul_1MatMul9gradients_dense_1_matmul_grad_matmul_1_model_mlp_fc3_tanh'gradients/Squeeze_grad/Reshape:output:0*
T0*
_output_shapes

:@*
transpose_a(�
*gradients/model/mlp_fc3/Tanh_grad/TanhGradTanhGrad9gradients_dense_1_matmul_grad_matmul_1_model_mlp_fc3_tanh.gradients/dense_1/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:	�@�
0gradients/model/mlp_fc3/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/mlp_fc3/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes
:@�
*gradients/model/mlp_fc3/MatMul_grad/MatMulMatMul.gradients/model/mlp_fc3/Tanh_grad/TanhGrad:z:0Ngradients_model_mlp_fc3_matmul_grad_matmul_model_mlp_fc3_matmul_readvariableop*
T0*
_output_shapes
:	�@*
transpose_b(�
,gradients/model/mlp_fc3/MatMul_grad/MatMul_1MatMul?gradients_model_mlp_fc3_matmul_grad_matmul_1_model_mlp_fc2_tanh.gradients/model/mlp_fc3/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
*gradients/model/mlp_fc2/Tanh_grad/TanhGradTanhGrad?gradients_model_mlp_fc3_matmul_grad_matmul_1_model_mlp_fc2_tanh4gradients/model/mlp_fc3/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:	�@�
0gradients/model/mlp_fc2/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/mlp_fc2/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes
:@�
*gradients/model/mlp_fc2/MatMul_grad/MatMulMatMul.gradients/model/mlp_fc2/Tanh_grad/TanhGrad:z:0Ngradients_model_mlp_fc2_matmul_grad_matmul_model_mlp_fc2_matmul_readvariableop*
T0*
_output_shapes
:	�@*
transpose_b(�
,gradients/model/mlp_fc2/MatMul_grad/MatMul_1MatMul?gradients_model_mlp_fc2_matmul_grad_matmul_1_model_mlp_fc1_tanh.gradients/model/mlp_fc2/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
*gradients/model/mlp_fc1/Tanh_grad/TanhGradTanhGrad?gradients_model_mlp_fc2_matmul_grad_matmul_1_model_mlp_fc1_tanh4gradients/model/mlp_fc2/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:	�@�
0gradients/model/mlp_fc1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/mlp_fc1/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes
:@�
*gradients/model/mlp_fc1/MatMul_grad/MatMulMatMul.gradients/model/mlp_fc1/Tanh_grad/TanhGrad:z:0Ngradients_model_mlp_fc1_matmul_grad_matmul_model_mlp_fc1_matmul_readvariableop*
T0*
_output_shapes
:	�@*
transpose_b(�
,gradients/model/mlp_fc1/MatMul_grad/MatMul_1MatMul?gradients_model_mlp_fc1_matmul_grad_matmul_1_model_mlp_fc0_tanh.gradients/model/mlp_fc1/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
*gradients/model/mlp_fc0/Tanh_grad/TanhGradTanhGrad?gradients_model_mlp_fc1_matmul_grad_matmul_1_model_mlp_fc0_tanh4gradients/model/mlp_fc1/MatMul_grad/MatMul:product:0*
T0*
_output_shapes
:	�@�
0gradients/model/mlp_fc0/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/model/mlp_fc0/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes
:@�
*gradients/model/mlp_fc0/MatMul_grad/MatMulMatMul.gradients/model/mlp_fc0/Tanh_grad/TanhGrad:z:0Ngradients_model_mlp_fc0_matmul_grad_matmul_model_mlp_fc0_matmul_readvariableop*
T0*
_output_shapes
:	�*
transpose_b(�
,gradients/model/mlp_fc0/MatMul_grad/MatMul_1MatMul7gradients_model_mlp_fc0_matmul_grad_matmul_1_model_cast.gradients/model/mlp_fc0/Tanh_grad/TanhGrad:z:0*
T0*
_output_shapes

:@*
transpose_a(u
IdentityIdentity6gradients/model/mlp_fc0/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@v

Identity_1Identity9gradients/model/mlp_fc0/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:@w

Identity_2Identity6gradients/model/mlp_fc1/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@@v

Identity_3Identity9gradients/model/mlp_fc1/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:@w

Identity_4Identity6gradients/model/mlp_fc2/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@@v

Identity_5Identity9gradients/model/mlp_fc2/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:@w

Identity_6Identity6gradients/model/mlp_fc3/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@@v

Identity_7Identity9gradients/model/mlp_fc3/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:@q

Identity_8Identity0gradients/dense_1/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@p

Identity_9Identity3gradients/dense_1/BiasAdd_grad/BiasAddGrad:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*�
_input_shapesr
p:�:@:	�@:@@:	�@:@@:	�@:@@:	�@:@:	�*0
forward_function_name__forward_value_52048:! 

_output_shapes	
:�:$ 

_output_shapes

:@:%!

_output_shapes
:	�@:$ 

_output_shapes

:@@:%!

_output_shapes
:	�@:$ 

_output_shapes

:@@:%!

_output_shapes
:	�@:$ 

_output_shapes

:@@:%!

_output_shapes
:	�@:$	 

_output_shapes

:@:%
!

_output_shapes
:	�
�
�
B__inference_model_layer_call_and_return_conditional_losses_1861443

inputs!
mlp_fc0_1861386:@
mlp_fc0_1861388:@!
mlp_fc1_1861403:@@
mlp_fc1_1861405:@!
mlp_fc2_1861420:@@
mlp_fc2_1861422:@!
mlp_fc3_1861437:@@
mlp_fc3_1861439:@
identity��mlp_fc0/StatefulPartitionedCall�mlp_fc1/StatefulPartitionedCall�mlp_fc2/StatefulPartitionedCall�mlp_fc3/StatefulPartitionedCall�
mlp_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsmlp_fc0_1861386mlp_fc0_1861388*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385�
mlp_fc1/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc0/StatefulPartitionedCall:output:0mlp_fc1_1861403mlp_fc1_1861405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402�
mlp_fc2/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc1/StatefulPartitionedCall:output:0mlp_fc2_1861420mlp_fc2_1861422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419�
mlp_fc3/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc2/StatefulPartitionedCall:output:0mlp_fc3_1861437mlp_fc3_1861439*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436w
IdentityIdentity(mlp_fc3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^mlp_fc0/StatefulPartitionedCall ^mlp_fc1/StatefulPartitionedCall ^mlp_fc2/StatefulPartitionedCall ^mlp_fc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
mlp_fc0/StatefulPartitionedCallmlp_fc0/StatefulPartitionedCall2B
mlp_fc1/StatefulPartitionedCallmlp_fc1/StatefulPartitionedCall2B
mlp_fc2/StatefulPartitionedCallmlp_fc2/StatefulPartitionedCall2B
mlp_fc3/StatefulPartitionedCallmlp_fc3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_mlp_fc3_layer_call_fn_1861812

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�/
�
__inference_value_52359
observation	>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOpX

model/CastCastobservation*

DstT0*

SrcT0	*
_output_shapes
:	��
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMulmodel/mlp_fc3/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�i
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
S
IdentityIdentitySqueeze:output:0^NoOp*
T0*
_output_shapes	
:��
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:	�: : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:L H

_output_shapes
:	�
%
_user_specified_nameobservation
�Y
�
 __inference__traced_save_1861967
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop(
$savev2_vf_kernel_read_readvariableop&
"savev2_vf_bias_read_readvariableop-
)savev2_mlp_fc0_kernel_read_readvariableop+
'savev2_mlp_fc0_bias_read_readvariableop-
)savev2_mlp_fc1_kernel_read_readvariableop+
'savev2_mlp_fc1_bias_read_readvariableop-
)savev2_mlp_fc2_kernel_read_readvariableop+
'savev2_mlp_fc2_bias_read_readvariableop-
)savev2_mlp_fc3_kernel_read_readvariableop+
'savev2_mlp_fc3_bias_read_readvariableop(
$savev2_pi_kernel_read_readvariableop&
"savev2_pi_bias_read_readvariableop/
+savev2_adam_vf_kernel_m_read_readvariableop-
)savev2_adam_vf_bias_m_read_readvariableop4
0savev2_adam_mlp_fc0_kernel_m_read_readvariableop2
.savev2_adam_mlp_fc0_bias_m_read_readvariableop4
0savev2_adam_mlp_fc1_kernel_m_read_readvariableop2
.savev2_adam_mlp_fc1_bias_m_read_readvariableop4
0savev2_adam_mlp_fc2_kernel_m_read_readvariableop2
.savev2_adam_mlp_fc2_bias_m_read_readvariableop4
0savev2_adam_mlp_fc3_kernel_m_read_readvariableop2
.savev2_adam_mlp_fc3_bias_m_read_readvariableop/
+savev2_adam_pi_kernel_m_read_readvariableop-
)savev2_adam_pi_bias_m_read_readvariableop/
+savev2_adam_vf_kernel_v_read_readvariableop-
)savev2_adam_vf_bias_v_read_readvariableop4
0savev2_adam_mlp_fc0_kernel_v_read_readvariableop2
.savev2_adam_mlp_fc0_bias_v_read_readvariableop4
0savev2_adam_mlp_fc1_kernel_v_read_readvariableop2
.savev2_adam_mlp_fc1_bias_v_read_readvariableop4
0savev2_adam_mlp_fc2_kernel_v_read_readvariableop2
.savev2_adam_mlp_fc2_bias_v_read_readvariableop4
0savev2_adam_mlp_fc3_kernel_v_read_readvariableop2
.savev2_adam_mlp_fc3_bias_v_read_readvariableop/
+savev2_adam_pi_kernel_v_read_readvariableop-
)savev2_adam_pi_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB6train_model/value_fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB4train_model/value_fc/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB@train_model/pdtype/matching_fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB>train_model/pdtype/matching_fc/bias/.ATTRIBUTES/VARIABLE_VALUEBRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop$savev2_vf_kernel_read_readvariableop"savev2_vf_bias_read_readvariableop)savev2_mlp_fc0_kernel_read_readvariableop'savev2_mlp_fc0_bias_read_readvariableop)savev2_mlp_fc1_kernel_read_readvariableop'savev2_mlp_fc1_bias_read_readvariableop)savev2_mlp_fc2_kernel_read_readvariableop'savev2_mlp_fc2_bias_read_readvariableop)savev2_mlp_fc3_kernel_read_readvariableop'savev2_mlp_fc3_bias_read_readvariableop$savev2_pi_kernel_read_readvariableop"savev2_pi_bias_read_readvariableop+savev2_adam_vf_kernel_m_read_readvariableop)savev2_adam_vf_bias_m_read_readvariableop0savev2_adam_mlp_fc0_kernel_m_read_readvariableop.savev2_adam_mlp_fc0_bias_m_read_readvariableop0savev2_adam_mlp_fc1_kernel_m_read_readvariableop.savev2_adam_mlp_fc1_bias_m_read_readvariableop0savev2_adam_mlp_fc2_kernel_m_read_readvariableop.savev2_adam_mlp_fc2_bias_m_read_readvariableop0savev2_adam_mlp_fc3_kernel_m_read_readvariableop.savev2_adam_mlp_fc3_bias_m_read_readvariableop+savev2_adam_pi_kernel_m_read_readvariableop)savev2_adam_pi_bias_m_read_readvariableop+savev2_adam_vf_kernel_v_read_readvariableop)savev2_adam_vf_bias_v_read_readvariableop0savev2_adam_mlp_fc0_kernel_v_read_readvariableop.savev2_adam_mlp_fc0_bias_v_read_readvariableop0savev2_adam_mlp_fc1_kernel_v_read_readvariableop.savev2_adam_mlp_fc1_bias_v_read_readvariableop0savev2_adam_mlp_fc2_kernel_v_read_readvariableop.savev2_adam_mlp_fc2_bias_v_read_readvariableop0savev2_adam_mlp_fc3_kernel_v_read_readvariableop.savev2_adam_mlp_fc3_bias_v_read_readvariableop+savev2_adam_pi_kernel_v_read_readvariableop)savev2_adam_pi_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : : : :@::@:@:@@:@:@@:@:@@:@:@::@::@:@:@@:@:@@:@:@@:@:@::@::@:@:@@:@:@@:@:@@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 	

_output_shapes
:@:$
 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::$ 

_output_shapes

:@: 

_output_shapes
::$  

_output_shapes

:@: !

_output_shapes
:@:$" 

_output_shapes

:@@: #

_output_shapes
:@:$$ 

_output_shapes

:@@: %

_output_shapes
:@:$& 

_output_shapes

:@@: '

_output_shapes
:@:$( 

_output_shapes

:@: )

_output_shapes
::*

_output_shapes
: 
�

�
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861803

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
ֈ
�
__inference_step_52279
observation	>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity	

identity_1

identity_2��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�&model/mlp_fc0/BiasAdd_1/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�%model/mlp_fc0/MatMul_1/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�&model/mlp_fc1/BiasAdd_1/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�%model/mlp_fc1/MatMul_1/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�&model/mlp_fc2/BiasAdd_1/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�%model/mlp_fc2/MatMul_1/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�&model/mlp_fc3/BiasAdd_1/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOp�%model/mlp_fc3/MatMul_1/ReadVariableOpW

model/CastCastobservation*

DstT0*

SrcT0	*
_output_shapes

:�
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes

:@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0|
dense/MatMulMatMulmodel/mlp_fc3/Tanh:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:V
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
random_uniform/RandomUniformRandomUniformShape:output:0*
T0*
_output_shapes

:*
dtype0*

seedZ
LogLog%random_uniform/RandomUniform:output:0*
T0*
_output_shapes

:<
NegNegLog:y:0*
T0*
_output_shapes

:>
Log_1LogNeg:y:0*
T0*
_output_shapes

:V
subSubdense/BiasAdd:output:0	Log_1:y:0*
T0*
_output_shapes

:[
ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������Y
ArgMaxArgMaxsub:z:0ArgMax/dimension:output:0*
T0*
_output_shapes
:U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
one_hotOneHotArgMax:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
_output_shapes

:h
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :x
'softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
%softmax_cross_entropy_with_logits/SubSub1softmax_cross_entropy_with_logits/Rank_1:output:00softmax_cross_entropy_with_logits/Sub/y:output:0*
T0*
_output_shapes
: �
-softmax_cross_entropy_with_logits/Slice/beginPack)softmax_cross_entropy_with_logits/Sub:z:0*
N*
T0*
_output_shapes
:v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
'softmax_cross_entropy_with_logits/SliceSlice2softmax_cross_entropy_with_logits/Shape_1:output:06softmax_cross_entropy_with_logits/Slice/begin:output:05softmax_cross_entropy_with_logits/Slice/size:output:0*
Index0*
T0*
_output_shapes
:�
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(softmax_cross_entropy_with_logits/concatConcatV2:softmax_cross_entropy_with_logits/concat/values_0:output:00softmax_cross_entropy_with_logits/Slice:output:06softmax_cross_entropy_with_logits/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/ReshapeReshapedense/BiasAdd:output:01softmax_cross_entropy_with_logits/concat:output:0*
T0*
_output_shapes

:j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_1Sub1softmax_cross_entropy_with_logits/Rank_2:output:02softmax_cross_entropy_with_logits/Sub_1/y:output:0*
T0*
_output_shapes
: �
/softmax_cross_entropy_with_logits/Slice_1/beginPack+softmax_cross_entropy_with_logits/Sub_1:z:0*
N*
T0*
_output_shapes
:x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
)softmax_cross_entropy_with_logits/Slice_1Slice2softmax_cross_entropy_with_logits/Shape_2:output:08softmax_cross_entropy_with_logits/Slice_1/begin:output:07softmax_cross_entropy_with_logits/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*softmax_cross_entropy_with_logits/concat_1ConcatV2<softmax_cross_entropy_with_logits/concat_1/values_0:output:02softmax_cross_entropy_with_logits/Slice_1:output:08softmax_cross_entropy_with_logits/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_1Reshapeone_hot:output:03softmax_cross_entropy_with_logits/concat_1:output:0*
T0*
_output_shapes

:�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2softmax_cross_entropy_with_logits/Reshape:output:04softmax_cross_entropy_with_logits/Reshape_1:output:0*
T0*$
_output_shapes
::k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_2Sub/softmax_cross_entropy_with_logits/Rank:output:02softmax_cross_entropy_with_logits/Sub_2/y:output:0*
T0*
_output_shapes
: y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: �
.softmax_cross_entropy_with_logits/Slice_2/sizePack+softmax_cross_entropy_with_logits/Sub_2:z:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/Slice_2Slice0softmax_cross_entropy_with_logits/Shape:output:08softmax_cross_entropy_with_logits/Slice_2/begin:output:07softmax_cross_entropy_with_logits/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_2Reshape(softmax_cross_entropy_with_logits:loss:02softmax_cross_entropy_with_logits/Slice_2:output:0*
T0*
_output_shapes
:Y
model/Cast_1Castobservation*

DstT0*

SrcT0	*
_output_shapes

:�
%model/mlp_fc0/MatMul_1/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMul_1MatMulmodel/Cast_1:y:0-model/mlp_fc0/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
&model/mlp_fc0/BiasAdd_1/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAdd_1BiasAdd model/mlp_fc0/MatMul_1:product:0.model/mlp_fc0/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
model/mlp_fc0/Tanh_1Tanh model/mlp_fc0/BiasAdd_1:output:0*
T0*
_output_shapes

:@�
%model/mlp_fc1/MatMul_1/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMul_1MatMulmodel/mlp_fc0/Tanh_1:y:0-model/mlp_fc1/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
&model/mlp_fc1/BiasAdd_1/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAdd_1BiasAdd model/mlp_fc1/MatMul_1:product:0.model/mlp_fc1/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
model/mlp_fc1/Tanh_1Tanh model/mlp_fc1/BiasAdd_1:output:0*
T0*
_output_shapes

:@�
%model/mlp_fc2/MatMul_1/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMul_1MatMulmodel/mlp_fc1/Tanh_1:y:0-model/mlp_fc2/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
&model/mlp_fc2/BiasAdd_1/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAdd_1BiasAdd model/mlp_fc2/MatMul_1:product:0.model/mlp_fc2/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
model/mlp_fc2/Tanh_1Tanh model/mlp_fc2/BiasAdd_1:output:0*
T0*
_output_shapes

:@�
%model/mlp_fc3/MatMul_1/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMul_1MatMulmodel/mlp_fc2/Tanh_1:y:0-model/mlp_fc3/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
&model/mlp_fc3/BiasAdd_1/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAdd_1BiasAdd model/mlp_fc3/MatMul_1:product:0.model/mlp_fc3/BiasAdd_1/ReadVariableOp:value:0*
T0*
_output_shapes

:@g
model/mlp_fc3/Tanh_1Tanh model/mlp_fc3/BiasAdd_1:output:0*
T0*
_output_shapes

:@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMulmodel/mlp_fc3/Tanh_1:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:h
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*
_output_shapes
:*
squeeze_dims
Q
IdentityIdentityArgMax:output:0^NoOp*
T0	*
_output_shapes
:T

Identity_1IdentitySqueeze:output:0^NoOp*
T0*
_output_shapes
:x

Identity_2Identity4softmax_cross_entropy_with_logits/Reshape_2:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp'^model/mlp_fc0/BiasAdd_1/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp&^model/mlp_fc0/MatMul_1/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp'^model/mlp_fc1/BiasAdd_1/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp&^model/mlp_fc1/MatMul_1/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp'^model/mlp_fc2/BiasAdd_1/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp&^model/mlp_fc2/MatMul_1/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp'^model/mlp_fc3/BiasAdd_1/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp&^model/mlp_fc3/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
":: : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2P
&model/mlp_fc0/BiasAdd_1/ReadVariableOp&model/mlp_fc0/BiasAdd_1/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2N
%model/mlp_fc0/MatMul_1/ReadVariableOp%model/mlp_fc0/MatMul_1/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2P
&model/mlp_fc1/BiasAdd_1/ReadVariableOp&model/mlp_fc1/BiasAdd_1/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2N
%model/mlp_fc1/MatMul_1/ReadVariableOp%model/mlp_fc1/MatMul_1/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2P
&model/mlp_fc2/BiasAdd_1/ReadVariableOp&model/mlp_fc2/BiasAdd_1/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2N
%model/mlp_fc2/MatMul_1/ReadVariableOp%model/mlp_fc2/MatMul_1/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2P
&model/mlp_fc3/BiasAdd_1/ReadVariableOp&model/mlp_fc3/BiasAdd_1/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp2N
%model/mlp_fc3/MatMul_1/ReadVariableOp%model/mlp_fc3/MatMul_1/ReadVariableOp:K G

_output_shapes

:
%
_user_specified_nameobservation
�'
�
"__inference__wrapped_model_1861367
input_1>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@
identity��$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOp�
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulinput_1+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@l
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@e
IdentityIdentitymodel/mlp_fc3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
��
�
#__inference__traced_restore_1862100
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: .
assignvariableop_5_vf_kernel:@(
assignvariableop_6_vf_bias:3
!assignvariableop_7_mlp_fc0_kernel:@-
assignvariableop_8_mlp_fc0_bias:@3
!assignvariableop_9_mlp_fc1_kernel:@@.
 assignvariableop_10_mlp_fc1_bias:@4
"assignvariableop_11_mlp_fc2_kernel:@@.
 assignvariableop_12_mlp_fc2_bias:@4
"assignvariableop_13_mlp_fc3_kernel:@@.
 assignvariableop_14_mlp_fc3_bias:@/
assignvariableop_15_pi_kernel:@)
assignvariableop_16_pi_bias:6
$assignvariableop_17_adam_vf_kernel_m:@0
"assignvariableop_18_adam_vf_bias_m:;
)assignvariableop_19_adam_mlp_fc0_kernel_m:@5
'assignvariableop_20_adam_mlp_fc0_bias_m:@;
)assignvariableop_21_adam_mlp_fc1_kernel_m:@@5
'assignvariableop_22_adam_mlp_fc1_bias_m:@;
)assignvariableop_23_adam_mlp_fc2_kernel_m:@@5
'assignvariableop_24_adam_mlp_fc2_bias_m:@;
)assignvariableop_25_adam_mlp_fc3_kernel_m:@@5
'assignvariableop_26_adam_mlp_fc3_bias_m:@6
$assignvariableop_27_adam_pi_kernel_m:@0
"assignvariableop_28_adam_pi_bias_m:6
$assignvariableop_29_adam_vf_kernel_v:@0
"assignvariableop_30_adam_vf_bias_v:;
)assignvariableop_31_adam_mlp_fc0_kernel_v:@5
'assignvariableop_32_adam_mlp_fc0_bias_v:@;
)assignvariableop_33_adam_mlp_fc1_kernel_v:@@5
'assignvariableop_34_adam_mlp_fc1_bias_v:@;
)assignvariableop_35_adam_mlp_fc2_kernel_v:@@5
'assignvariableop_36_adam_mlp_fc2_bias_v:@;
)assignvariableop_37_adam_mlp_fc3_kernel_v:@@5
'assignvariableop_38_adam_mlp_fc3_bias_v:@6
$assignvariableop_39_adam_pi_kernel_v:@0
"assignvariableop_40_adam_pi_bias_v:
identity_42��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*�
value�B�*B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB6train_model/value_fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB4train_model/value_fc/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEBQtrain_model/policy_network/layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEBOtrain_model/policy_network/layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB@train_model/pdtype/matching_fc/kernel/.ATTRIBUTES/VARIABLE_VALUEB>train_model/pdtype/matching_fc/bias/.ATTRIBUTES/VARIABLE_VALUEBRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEB\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRtrain_model/value_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPtrain_model/value_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBmtrain_model/policy_network/layer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBktrain_model/policy_network/layer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB\train_model/pdtype/matching_fc/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBZtrain_model/pdtype/matching_fc/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_vf_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_vf_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_mlp_fc0_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_mlp_fc0_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp!assignvariableop_9_mlp_fc1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp assignvariableop_10_mlp_fc1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp"assignvariableop_11_mlp_fc2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp assignvariableop_12_mlp_fc2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_mlp_fc3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp assignvariableop_14_mlp_fc3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_pi_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_pi_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp$assignvariableop_17_adam_vf_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp"assignvariableop_18_adam_vf_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_mlp_fc0_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_mlp_fc0_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_adam_mlp_fc1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_adam_mlp_fc1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_mlp_fc2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_mlp_fc2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_mlp_fc3_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_mlp_fc3_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp$assignvariableop_27_adam_pi_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp"assignvariableop_28_adam_pi_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp$assignvariableop_29_adam_vf_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp"assignvariableop_30_adam_vf_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_mlp_fc0_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_mlp_fc0_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_mlp_fc1_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_mlp_fc1_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_mlp_fc2_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_mlp_fc2_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_mlp_fc3_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_mlp_fc3_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp$assignvariableop_39_adam_pi_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp"assignvariableop_40_adam_pi_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_42IdentityIdentity_41:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
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
�#
�
B__inference_model_layer_call_and_return_conditional_losses_1861711

inputs8
&mlp_fc0_matmul_readvariableop_resource:@5
'mlp_fc0_biasadd_readvariableop_resource:@8
&mlp_fc1_matmul_readvariableop_resource:@@5
'mlp_fc1_biasadd_readvariableop_resource:@8
&mlp_fc2_matmul_readvariableop_resource:@@5
'mlp_fc2_biasadd_readvariableop_resource:@8
&mlp_fc3_matmul_readvariableop_resource:@@5
'mlp_fc3_biasadd_readvariableop_resource:@
identity��mlp_fc0/BiasAdd/ReadVariableOp�mlp_fc0/MatMul/ReadVariableOp�mlp_fc1/BiasAdd/ReadVariableOp�mlp_fc1/MatMul/ReadVariableOp�mlp_fc2/BiasAdd/ReadVariableOp�mlp_fc2/MatMul/ReadVariableOp�mlp_fc3/BiasAdd/ReadVariableOp�mlp_fc3/MatMul/ReadVariableOp�
mlp_fc0/MatMul/ReadVariableOpReadVariableOp&mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
mlp_fc0/MatMulMatMulinputs%mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc0/BiasAddBiasAddmlp_fc0/MatMul:product:0&mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc0/TanhTanhmlp_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc1/MatMul/ReadVariableOpReadVariableOp&mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc1/MatMulMatMulmlp_fc0/Tanh:y:0%mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc1/BiasAddBiasAddmlp_fc1/MatMul:product:0&mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc1/TanhTanhmlp_fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc2/MatMul/ReadVariableOpReadVariableOp&mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc2/MatMulMatMulmlp_fc1/Tanh:y:0%mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc2/BiasAddBiasAddmlp_fc2/MatMul:product:0&mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc2/TanhTanhmlp_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc3/MatMul/ReadVariableOpReadVariableOp&mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc3/MatMulMatMulmlp_fc2/Tanh:y:0%mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc3/BiasAddBiasAddmlp_fc3/MatMul:product:0&mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc3/TanhTanhmlp_fc3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@_
IdentityIdentitymlp_fc3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^mlp_fc0/BiasAdd/ReadVariableOp^mlp_fc0/MatMul/ReadVariableOp^mlp_fc1/BiasAdd/ReadVariableOp^mlp_fc1/MatMul/ReadVariableOp^mlp_fc2/BiasAdd/ReadVariableOp^mlp_fc2/MatMul/ReadVariableOp^mlp_fc3/BiasAdd/ReadVariableOp^mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2@
mlp_fc0/BiasAdd/ReadVariableOpmlp_fc0/BiasAdd/ReadVariableOp2>
mlp_fc0/MatMul/ReadVariableOpmlp_fc0/MatMul/ReadVariableOp2@
mlp_fc1/BiasAdd/ReadVariableOpmlp_fc1/BiasAdd/ReadVariableOp2>
mlp_fc1/MatMul/ReadVariableOpmlp_fc1/MatMul/ReadVariableOp2@
mlp_fc2/BiasAdd/ReadVariableOpmlp_fc2/BiasAdd/ReadVariableOp2>
mlp_fc2/MatMul/ReadVariableOpmlp_fc2/MatMul/ReadVariableOp2@
mlp_fc3/BiasAdd/ReadVariableOpmlp_fc3/BiasAdd/ReadVariableOp2>
mlp_fc3/MatMul/ReadVariableOpmlp_fc3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861823

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
'__inference_model_layer_call_fn_1861679

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1861549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
)__inference_mlp_fc0_layer_call_fn_1861752

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�#
�
B__inference_model_layer_call_and_return_conditional_losses_1861743

inputs8
&mlp_fc0_matmul_readvariableop_resource:@5
'mlp_fc0_biasadd_readvariableop_resource:@8
&mlp_fc1_matmul_readvariableop_resource:@@5
'mlp_fc1_biasadd_readvariableop_resource:@8
&mlp_fc2_matmul_readvariableop_resource:@@5
'mlp_fc2_biasadd_readvariableop_resource:@8
&mlp_fc3_matmul_readvariableop_resource:@@5
'mlp_fc3_biasadd_readvariableop_resource:@
identity��mlp_fc0/BiasAdd/ReadVariableOp�mlp_fc0/MatMul/ReadVariableOp�mlp_fc1/BiasAdd/ReadVariableOp�mlp_fc1/MatMul/ReadVariableOp�mlp_fc2/BiasAdd/ReadVariableOp�mlp_fc2/MatMul/ReadVariableOp�mlp_fc3/BiasAdd/ReadVariableOp�mlp_fc3/MatMul/ReadVariableOp�
mlp_fc0/MatMul/ReadVariableOpReadVariableOp&mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0y
mlp_fc0/MatMulMatMulinputs%mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc0/BiasAddBiasAddmlp_fc0/MatMul:product:0&mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc0/TanhTanhmlp_fc0/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc1/MatMul/ReadVariableOpReadVariableOp&mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc1/MatMulMatMulmlp_fc0/Tanh:y:0%mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc1/BiasAddBiasAddmlp_fc1/MatMul:product:0&mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc1/TanhTanhmlp_fc1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc2/MatMul/ReadVariableOpReadVariableOp&mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc2/MatMulMatMulmlp_fc1/Tanh:y:0%mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc2/BiasAddBiasAddmlp_fc2/MatMul:product:0&mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc2/TanhTanhmlp_fc2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
mlp_fc3/MatMul/ReadVariableOpReadVariableOp&mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
mlp_fc3/MatMulMatMulmlp_fc2/Tanh:y:0%mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp'mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
mlp_fc3/BiasAddBiasAddmlp_fc3/MatMul:product:0&mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@`
mlp_fc3/TanhTanhmlp_fc3/BiasAdd:output:0*
T0*'
_output_shapes
:���������@_
IdentityIdentitymlp_fc3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^mlp_fc0/BiasAdd/ReadVariableOp^mlp_fc0/MatMul/ReadVariableOp^mlp_fc1/BiasAdd/ReadVariableOp^mlp_fc1/MatMul/ReadVariableOp^mlp_fc2/BiasAdd/ReadVariableOp^mlp_fc2/MatMul/ReadVariableOp^mlp_fc3/BiasAdd/ReadVariableOp^mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2@
mlp_fc0/BiasAdd/ReadVariableOpmlp_fc0/BiasAdd/ReadVariableOp2>
mlp_fc0/MatMul/ReadVariableOpmlp_fc0/MatMul/ReadVariableOp2@
mlp_fc1/BiasAdd/ReadVariableOpmlp_fc1/BiasAdd/ReadVariableOp2>
mlp_fc1/MatMul/ReadVariableOpmlp_fc1/MatMul/ReadVariableOp2@
mlp_fc2/BiasAdd/ReadVariableOpmlp_fc2/BiasAdd/ReadVariableOp2>
mlp_fc2/MatMul/ReadVariableOpmlp_fc2/MatMul/ReadVariableOp2@
mlp_fc3/BiasAdd/ReadVariableOpmlp_fc3/BiasAdd/ReadVariableOp2>
mlp_fc3/MatMul/ReadVariableOpmlp_fc3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
'__inference_model_layer_call_fn_1861658

inputs
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1861443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1861549

inputs!
mlp_fc0_1861528:@
mlp_fc0_1861530:@!
mlp_fc1_1861533:@@
mlp_fc1_1861535:@!
mlp_fc2_1861538:@@
mlp_fc2_1861540:@!
mlp_fc3_1861543:@@
mlp_fc3_1861545:@
identity��mlp_fc0/StatefulPartitionedCall�mlp_fc1/StatefulPartitionedCall�mlp_fc2/StatefulPartitionedCall�mlp_fc3/StatefulPartitionedCall�
mlp_fc0/StatefulPartitionedCallStatefulPartitionedCallinputsmlp_fc0_1861528mlp_fc0_1861530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385�
mlp_fc1/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc0/StatefulPartitionedCall:output:0mlp_fc1_1861533mlp_fc1_1861535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402�
mlp_fc2/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc1/StatefulPartitionedCall:output:0mlp_fc2_1861538mlp_fc2_1861540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419�
mlp_fc3/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc2/StatefulPartitionedCall:output:0mlp_fc3_1861543mlp_fc3_1861545*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436w
IdentityIdentity(mlp_fc3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^mlp_fc0/StatefulPartitionedCall ^mlp_fc1/StatefulPartitionedCall ^mlp_fc2/StatefulPartitionedCall ^mlp_fc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
mlp_fc0/StatefulPartitionedCallmlp_fc0/StatefulPartitionedCall2B
mlp_fc1/StatefulPartitionedCallmlp_fc1/StatefulPartitionedCall2B
mlp_fc2/StatefulPartitionedCallmlp_fc2/StatefulPartitionedCall2B
mlp_fc3/StatefulPartitionedCallmlp_fc3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861783

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
)__inference_mlp_fc1_layer_call_fn_1861772

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1861613
input_1!
mlp_fc0_1861592:@
mlp_fc0_1861594:@!
mlp_fc1_1861597:@@
mlp_fc1_1861599:@!
mlp_fc2_1861602:@@
mlp_fc2_1861604:@!
mlp_fc3_1861607:@@
mlp_fc3_1861609:@
identity��mlp_fc0/StatefulPartitionedCall�mlp_fc1/StatefulPartitionedCall�mlp_fc2/StatefulPartitionedCall�mlp_fc3/StatefulPartitionedCall�
mlp_fc0/StatefulPartitionedCallStatefulPartitionedCallinput_1mlp_fc0_1861592mlp_fc0_1861594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385�
mlp_fc1/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc0/StatefulPartitionedCall:output:0mlp_fc1_1861597mlp_fc1_1861599*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402�
mlp_fc2/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc1/StatefulPartitionedCall:output:0mlp_fc2_1861602mlp_fc2_1861604*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419�
mlp_fc3/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc2/StatefulPartitionedCall:output:0mlp_fc3_1861607mlp_fc3_1861609*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436w
IdentityIdentity(mlp_fc3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^mlp_fc0/StatefulPartitionedCall ^mlp_fc1/StatefulPartitionedCall ^mlp_fc2/StatefulPartitionedCall ^mlp_fc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
mlp_fc0/StatefulPartitionedCallmlp_fc0/StatefulPartitionedCall2B
mlp_fc1/StatefulPartitionedCallmlp_fc1/StatefulPartitionedCall2B
mlp_fc2/StatefulPartitionedCallmlp_fc2/StatefulPartitionedCall2B
mlp_fc3/StatefulPartitionedCallmlp_fc3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_1861637
input_1!
mlp_fc0_1861616:@
mlp_fc0_1861618:@!
mlp_fc1_1861621:@@
mlp_fc1_1861623:@!
mlp_fc2_1861626:@@
mlp_fc2_1861628:@!
mlp_fc3_1861631:@@
mlp_fc3_1861633:@
identity��mlp_fc0/StatefulPartitionedCall�mlp_fc1/StatefulPartitionedCall�mlp_fc2/StatefulPartitionedCall�mlp_fc3/StatefulPartitionedCall�
mlp_fc0/StatefulPartitionedCallStatefulPartitionedCallinput_1mlp_fc0_1861616mlp_fc0_1861618*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861385�
mlp_fc1/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc0/StatefulPartitionedCall:output:0mlp_fc1_1861621mlp_fc1_1861623*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861402�
mlp_fc2/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc1/StatefulPartitionedCall:output:0mlp_fc2_1861626mlp_fc2_1861628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861419�
mlp_fc3/StatefulPartitionedCallStatefulPartitionedCall(mlp_fc2/StatefulPartitionedCall:output:0mlp_fc3_1861631mlp_fc3_1861633*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *M
fHRF
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861436w
IdentityIdentity(mlp_fc3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp ^mlp_fc0/StatefulPartitionedCall ^mlp_fc1/StatefulPartitionedCall ^mlp_fc2/StatefulPartitionedCall ^mlp_fc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 2B
mlp_fc0/StatefulPartitionedCallmlp_fc0/StatefulPartitionedCall2B
mlp_fc1/StatefulPartitionedCallmlp_fc1/StatefulPartitionedCall2B
mlp_fc2/StatefulPartitionedCallmlp_fc2/StatefulPartitionedCall2B
mlp_fc3/StatefulPartitionedCallmlp_fc3/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
'__inference_model_layer_call_fn_1861589
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1861549o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
'__inference_model_layer_call_fn_1861462
input_1
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_1861443o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�/
�
__inference_value_51792
observation	>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOpX

model/CastCastobservation*

DstT0*

SrcT0	*
_output_shapes
:	��
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMulmodel/mlp_fc3/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�i
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
S
IdentityIdentitySqueeze:output:0^NoOp*
T0*
_output_shapes	
:��
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:	�: : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:L H

_output_shapes
:	�
%
_user_specified_nameobservation
��
�
__inference_get_grad_52166
obs	
returns	
masks

actions	

values
neglogpac_old>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:
unknown:@
	unknown_0:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8

identity_9
identity_10
identity_11
identity_12
identity_13
identity_14
identity_15
identity_16��StatefulPartitionedCall�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOpA
subSubreturnsvalues*
T0*
_output_shapes	
:�O
ConstConst*
_output_shapes
:*
dtype0*
valueB: F
MeanMeansub:z:0Const:output:0*
T0*
_output_shapes
: J
sub_1Subsub:z:0Mean:output:0*
T0*
_output_shapes	
:�j
 reduce_std/reduce_variance/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
reduce_std/reduce_variance/MeanMeansub:z:0)reduce_std/reduce_variance/Const:output:0*
T0*
_output_shapes
:*
	keep_dims(~
reduce_std/reduce_variance/subSubsub:z:0(reduce_std/reduce_variance/Mean:output:0*
T0*
_output_shapes	
:�u
!reduce_std/reduce_variance/SquareSquare"reduce_std/reduce_variance/sub:z:0*
T0*
_output_shapes	
:�l
"reduce_std/reduce_variance/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
!reduce_std/reduce_variance/Mean_1Mean%reduce_std/reduce_variance/Square:y:0+reduce_std/reduce_variance/Const_1:output:0*
T0*
_output_shapes
: d
reduce_std/SqrtSqrt*reduce_std/reduce_variance/Mean_1:output:0*
T0*
_output_shapes
: J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *w�+2R
addAddV2reduce_std/Sqrt:y:0add/y:output:0*
T0*
_output_shapes
: L
truedivRealDiv	sub_1:z:0add:z:0*
T0*
_output_shapes	
:�P

model/CastCastobs*

DstT0*

SrcT0	*
_output_shapes
:	��
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0}
dense/MatMulMatMulmodel/mlp_fc3/Tanh:y:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�U
one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  �?V
one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    O
one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :�
one_hotOneHotactionsone_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
_output_shapes
:	�h
&softmax_cross_entropy_with_logits/RankConst*
_output_shapes
: *
dtype0*
value	B :x
'softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      j
(softmax_cross_entropy_with_logits/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      i
'softmax_cross_entropy_with_logits/Sub/yConst*
_output_shapes
: *
dtype0*
value	B :�
%softmax_cross_entropy_with_logits/SubSub1softmax_cross_entropy_with_logits/Rank_1:output:00softmax_cross_entropy_with_logits/Sub/y:output:0*
T0*
_output_shapes
: �
-softmax_cross_entropy_with_logits/Slice/beginPack)softmax_cross_entropy_with_logits/Sub:z:0*
N*
T0*
_output_shapes
:v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
'softmax_cross_entropy_with_logits/SliceSlice2softmax_cross_entropy_with_logits/Shape_1:output:06softmax_cross_entropy_with_logits/Slice/begin:output:05softmax_cross_entropy_with_logits/Slice/size:output:0*
Index0*
T0*
_output_shapes
:�
1softmax_cross_entropy_with_logits/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������o
-softmax_cross_entropy_with_logits/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
(softmax_cross_entropy_with_logits/concatConcatV2:softmax_cross_entropy_with_logits/concat/values_0:output:00softmax_cross_entropy_with_logits/Slice:output:06softmax_cross_entropy_with_logits/concat/axis:output:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/ReshapeReshapedense/BiasAdd:output:01softmax_cross_entropy_with_logits/concat:output:0*
T0*
_output_shapes
:	�j
(softmax_cross_entropy_with_logits/Rank_2Const*
_output_shapes
: *
dtype0*
value	B :z
)softmax_cross_entropy_with_logits/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"      k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_1Sub1softmax_cross_entropy_with_logits/Rank_2:output:02softmax_cross_entropy_with_logits/Sub_1/y:output:0*
T0*
_output_shapes
: �
/softmax_cross_entropy_with_logits/Slice_1/beginPack+softmax_cross_entropy_with_logits/Sub_1:z:0*
N*
T0*
_output_shapes
:x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
_output_shapes
:*
dtype0*
valueB:�
)softmax_cross_entropy_with_logits/Slice_1Slice2softmax_cross_entropy_with_logits/Shape_2:output:08softmax_cross_entropy_with_logits/Slice_1/begin:output:07softmax_cross_entropy_with_logits/Slice_1/size:output:0*
Index0*
T0*
_output_shapes
:�
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB:
���������q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
*softmax_cross_entropy_with_logits/concat_1ConcatV2<softmax_cross_entropy_with_logits/concat_1/values_0:output:02softmax_cross_entropy_with_logits/Slice_1:output:08softmax_cross_entropy_with_logits/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_1Reshapeone_hot:output:03softmax_cross_entropy_with_logits/concat_1:output:0*
T0*
_output_shapes
:	��
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits2softmax_cross_entropy_with_logits/Reshape:output:04softmax_cross_entropy_with_logits/Reshape_1:output:0*
T0*&
_output_shapes
:�:	�k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
_output_shapes
: *
dtype0*
value	B :�
'softmax_cross_entropy_with_logits/Sub_2Sub/softmax_cross_entropy_with_logits/Rank:output:02softmax_cross_entropy_with_logits/Sub_2/y:output:0*
T0*
_output_shapes
: y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
_output_shapes
:*
dtype0*
valueB: �
.softmax_cross_entropy_with_logits/Slice_2/sizePack+softmax_cross_entropy_with_logits/Sub_2:z:0*
N*
T0*
_output_shapes
:�
)softmax_cross_entropy_with_logits/Slice_2Slice0softmax_cross_entropy_with_logits/Shape:output:08softmax_cross_entropy_with_logits/Slice_2/begin:output:07softmax_cross_entropy_with_logits/Slice_2/size:output:0*
Index0*
T0*
_output_shapes
:�
+softmax_cross_entropy_with_logits/Reshape_2Reshape(softmax_cross_entropy_with_logits:loss:02softmax_cross_entropy_with_logits/Slice_2:output:0*
T0*
_output_shapes	
:�`
Max/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������}
MaxMaxdense/BiasAdd:output:0Max/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(\
sub_2Subdense/BiasAdd:output:0Max:output:0*
T0*
_output_shapes
:	�?
ExpExp	sub_2:z:0*
T0*
_output_shapes
:	�`
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������n
SumSumExp:y:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(U
	truediv_1RealDivExp:y:0Sum:output:0*
T0*
_output_shapes
:	�B
LogLogSum:output:0*
T0*
_output_shapes
:	�J
sub_3SubLog:y:0	sub_2:z:0*
T0*
_output_shapes
:	�N
mulMultruediv_1:z:0	sub_3:z:0*
T0*
_output_shapes
:	�b
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
���������]
Sum_1Summul:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes	
:�Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: Q
Mean_1MeanSum_1:output:0Const_1:output:0*
T0*
_output_shapes
: �
StatefulPartitionedCallStatefulPartitionedCallobs,model_mlp_fc0_matmul_readvariableop_resource-model_mlp_fc0_biasadd_readvariableop_resource,model_mlp_fc1_matmul_readvariableop_resource-model_mlp_fc1_biasadd_readvariableop_resource,model_mlp_fc2_matmul_readvariableop_resource-model_mlp_fc2_biasadd_readvariableop_resource,model_mlp_fc3_matmul_readvariableop_resource-model_mlp_fc3_biasadd_readvariableop_resourceunknown	unknown_0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *�
_output_shapesr
p:�:@:	�@:@@:	�@:@@:	�@:@@:	�@:@:	�*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__forward_value_52048\
sub_4Sub StatefulPartitionedCall:output:0values*
T0*
_output_shapes	
:�\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>s
clip_by_value/MinimumMinimum	sub_4:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes	
:�T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L�s
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes	
:�O
add_1AddV2valuesclip_by_value:z:0*
T0*
_output_shapes	
:�]
sub_5Sub StatefulPartitionedCall:output:0returns*
T0*
_output_shapes	
:�A
SquareSquare	sub_5:z:0*
T0*
_output_shapes	
:�F
sub_6Sub	add_1:z:0returns*
T0*
_output_shapes	
:�C
Square_1Square	sub_6:z:0*
T0*
_output_shapes	
:�R
MaximumMaximum
Square:y:0Square_1:y:0*
T0*
_output_shapes	
:�Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB: N
Mean_2MeanMaximum:z:0Const_2:output:0*
T0*
_output_shapes
: L
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?P
mul_1Mulmul_1/x:output:0Mean_2:output:0*
T0*
_output_shapes
: w
sub_7Subneglogpac_old4softmax_cross_entropy_with_logits/Reshape_2:output:0*
T0*
_output_shapes	
:�=
Exp_1Exp	sub_7:z:0*
T0*
_output_shapes	
:�=
NegNegtruediv:z:0*
T0*
_output_shapes	
:�F
mul_2MulNeg:y:0	Exp_1:y:0*
T0*
_output_shapes	
:�?
Neg_1Negtruediv:z:0*
T0*
_output_shapes	
:�^
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���?w
clip_by_value_1/MinimumMinimum	Exp_1:y:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes	
:�V
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L?y
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes	
:�R
mul_3Mul	Neg_1:y:0clip_by_value_1:z:0*
T0*
_output_shapes	
:�P
	Maximum_1Maximum	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes	
:�Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: P
Mean_3MeanMaximum_1:z:0Const_3:output:0*
T0*
_output_shapes
: w
sub_8Sub4softmax_cross_entropy_with_logits/Reshape_2:output:0neglogpac_old*
T0*
_output_shapes	
:�C
Square_2Square	sub_8:z:0*
T0*
_output_shapes	
:�Q
Const_4Const*
_output_shapes
:*
dtype0*
valueB: O
Mean_4MeanSquare_2:y:0Const_4:output:0*
T0*
_output_shapes
: L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?P
mul_4Mulmul_4/x:output:0Mean_4:output:0*
T0*
_output_shapes
: L
sub_9/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �?O
sub_9Sub	Exp_1:y:0sub_9/y:output:0*
T0*
_output_shapes	
:�;
AbsAbs	sub_9:z:0*
T0*
_output_shapes	
:�N
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>U
GreaterGreaterAbs:y:0Greater/y:output:0*
T0*
_output_shapes	
:�N
CastCastGreater:z:0*

DstT0*

SrcT0
*
_output_shapes	
:�Q
Const_5Const*
_output_shapes
:*
dtype0*
valueB: K
Mean_5MeanCast:y:0Const_5:output:0*
T0*
_output_shapes
: L
mul_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *    P
mul_5MulMean_1:output:0mul_5/y:output:0*
T0*
_output_shapes
: J
sub_10SubMean_3:output:0	mul_5:z:0*
T0*
_output_shapes
: L
mul_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?J
mul_6Mul	mul_1:z:0mul_6/y:output:0*
T0*
_output_shapes
: F
add_2AddV2
sub_10:z:0	mul_6:z:0*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?O
gradient_tape/sub_10/NegNegones:output:0*
T0*
_output_shapes
: `
gradient_tape/mul_6/MulMulones:output:0mul_6/y:output:0*
T0*
_output_shapes
: e
gradient_tape/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:z
gradient_tape/ReshapeReshapeones:output:0$gradient_tape/Reshape/shape:output:0*
T0*
_output_shapes
:^
gradient_tape/ConstConst*
_output_shapes
:*
dtype0*
valueB:�~
gradient_tape/TileTilegradient_tape/Reshape:output:0gradient_tape/Const:output:0*
T0*
_output_shapes	
:�Z
gradient_tape/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  �C�
gradient_tape/truedivRealDivgradient_tape/Tile:output:0gradient_tape/Const_1:output:0*
T0*
_output_shapes	
:�o
gradient_tape/mul_5/MulMulgradient_tape/sub_10/Neg:y:0mul_5/y:output:0*
T0*
_output_shapes
: m
gradient_tape/mul_1/MulMulgradient_tape/mul_6/Mul:z:0Mean_2:output:0*
T0*
_output_shapes
: p
gradient_tape/mul_1/Mul_1Mulgradient_tape/mul_6/Mul:z:0mul_1/x:output:0*
T0*
_output_shapes
: ^
gradient_tape/ShapeConst*
_output_shapes
:*
dtype0*
valueB:�`
gradient_tape/Shape_1Const*
_output_shapes
:*
dtype0*
valueB:�g
gradient_tape/zeros_likeConst*
_output_shapes	
:�*
dtype0*
valueB�*    f
gradient_tape/GreaterEqualGreaterEqual	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes	
:��
#gradient_tape/BroadcastGradientArgsBroadcastGradientArgsgradient_tape/Shape:output:0gradient_tape/Shape_1:output:0*2
_output_shapes 
:���������:����������
gradient_tape/SelectV2SelectV2gradient_tape/GreaterEqual:z:0gradient_tape/truediv:z:0!gradient_tape/zeros_like:output:0*
T0*
_output_shapes	
:��
gradient_tape/SumSumgradient_tape/SelectV2:output:0(gradient_tape/BroadcastGradientArgs:r0:0*
T0*
_output_shapes	
:��
gradient_tape/Reshape_1Reshapegradient_tape/Sum:output:0gradient_tape/Shape:output:0*
T0*
_output_shapes	
:��
gradient_tape/SelectV2_1SelectV2gradient_tape/GreaterEqual:z:0!gradient_tape/zeros_like:output:0gradient_tape/truediv:z:0*
T0*
_output_shapes	
:��
gradient_tape/Sum_1Sum!gradient_tape/SelectV2_1:output:0(gradient_tape/BroadcastGradientArgs:r1:0*
T0*
_output_shapes	
:��
gradient_tape/Reshape_2Reshapegradient_tape/Sum_1:output:0gradient_tape/Shape_1:output:0*
T0*
_output_shapes	
:�g
gradient_tape/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
gradient_tape/Reshape_3Reshapegradient_tape/mul_5/Mul:z:0&gradient_tape/Reshape_3/shape:output:0*
T0*
_output_shapes
:`
gradient_tape/Const_2Const*
_output_shapes
:*
dtype0*
valueB:��
gradient_tape/Tile_1Tile gradient_tape/Reshape_3:output:0gradient_tape/Const_2:output:0*
T0*
_output_shapes	
:�Z
gradient_tape/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  �C�
gradient_tape/truediv_1RealDivgradient_tape/Tile_1:output:0gradient_tape/Const_3:output:0*
T0*
_output_shapes	
:�g
gradient_tape/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
gradient_tape/Reshape_4Reshapegradient_tape/mul_1/Mul_1:z:0&gradient_tape/Reshape_4/shape:output:0*
T0*
_output_shapes
:`
gradient_tape/Const_4Const*
_output_shapes
:*
dtype0*
valueB:��
gradient_tape/Tile_2Tile gradient_tape/Reshape_4:output:0gradient_tape/Const_4:output:0*
T0*
_output_shapes	
:�Z
gradient_tape/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  �C�
gradient_tape/truediv_2RealDivgradient_tape/Tile_2:output:0gradient_tape/Const_5:output:0*
T0*
_output_shapes	
:�q
gradient_tape/mul_2/MulMul gradient_tape/Reshape_1:output:0	Exp_1:y:0*
T0*
_output_shapes	
:�q
gradient_tape/mul_2/Mul_1Mul gradient_tape/Reshape_1:output:0Neg:y:0*
T0*
_output_shapes	
:�{
gradient_tape/mul_3/MulMul gradient_tape/Reshape_2:output:0clip_by_value_1:z:0*
T0*
_output_shapes	
:�s
gradient_tape/mul_3/Mul_1Mul gradient_tape/Reshape_2:output:0	Neg_1:y:0*
T0*
_output_shapes	
:�h
gradient_tape/Maximum/xConst*
_output_shapes
:*
dtype0*
valueB"      Y
gradient_tape/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
gradient_tape/MaximumMaximum gradient_tape/Maximum/x:output:0 gradient_tape/Maximum/y:output:0*
T0*
_output_shapes
:i
gradient_tape/floordiv/xConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/floordivFloorDiv!gradient_tape/floordiv/x:output:0gradient_tape/Maximum:z:0*
T0*
_output_shapes
:n
gradient_tape/Reshape_5/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/Reshape_5Reshapegradient_tape/truediv_1:z:0&gradient_tape/Reshape_5/shape:output:0*
T0*
_output_shapes
:	�o
gradient_tape/Tile_3/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/Tile_3Tile gradient_tape/Reshape_5:output:0'gradient_tape/Tile_3/multiples:output:0*
T0*
_output_shapes
:	�`
gradient_tape/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:�`
gradient_tape/Shape_3Const*
_output_shapes
:*
dtype0*
valueB:�i
gradient_tape/zeros_like_1Const*
_output_shapes	
:�*
dtype0*
valueB�*    l
gradient_tape/GreaterEqual_1GreaterEqual
Square:y:0Square_1:y:0*
T0*
_output_shapes	
:��
%gradient_tape/BroadcastGradientArgs_1BroadcastGradientArgsgradient_tape/Shape_2:output:0gradient_tape/Shape_3:output:0*2
_output_shapes 
:���������:����������
gradient_tape/SelectV2_2SelectV2 gradient_tape/GreaterEqual_1:z:0gradient_tape/truediv_2:z:0#gradient_tape/zeros_like_1:output:0*
T0*
_output_shapes	
:��
gradient_tape/Sum_2Sum!gradient_tape/SelectV2_2:output:0*gradient_tape/BroadcastGradientArgs_1:r0:0*
T0*
_output_shapes	
:��
gradient_tape/Reshape_6Reshapegradient_tape/Sum_2:output:0gradient_tape/Shape_2:output:0*
T0*
_output_shapes	
:��
gradient_tape/SelectV2_3SelectV2 gradient_tape/GreaterEqual_1:z:0#gradient_tape/zeros_like_1:output:0gradient_tape/truediv_2:z:0*
T0*
_output_shapes	
:��
gradient_tape/Sum_3Sum!gradient_tape/SelectV2_3:output:0*gradient_tape/BroadcastGradientArgs_1:r1:0*
T0*
_output_shapes	
:��
gradient_tape/Reshape_7Reshapegradient_tape/Sum_3:output:0gradient_tape/Shape_3:output:0*
T0*
_output_shapes	
:�w
(gradient_tape/clip_by_value_1/zeros_likeConst*
_output_shapes	
:�*
dtype0*
valueB�*    �
*gradient_tape/clip_by_value_1/GreaterEqualGreaterEqualclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*
_output_shapes	
:��
&gradient_tape/clip_by_value_1/SelectV2SelectV2.gradient_tape/clip_by_value_1/GreaterEqual:z:0gradient_tape/mul_3/Mul_1:z:01gradient_tape/clip_by_value_1/zeros_like:output:0*
T0*
_output_shapes	
:�p
gradient_tape/mul/MulMulgradient_tape/Tile_3:output:0	sub_3:z:0*
T0*
_output_shapes
:	�v
gradient_tape/mul/Mul_1Mulgradient_tape/Tile_3:output:0truediv_1:z:0*
T0*
_output_shapes
:	�t
gradient_tape/Const_6Const^gradient_tape/Reshape_6*
_output_shapes
: *
dtype0*
valueB
 *   @i
gradient_tape/MulMul	sub_5:z:0gradient_tape/Const_6:output:0*
T0*
_output_shapes	
:�y
gradient_tape/Mul_1Mul gradient_tape/Reshape_6:output:0gradient_tape/Mul:z:0*
T0*
_output_shapes	
:�t
gradient_tape/Const_7Const^gradient_tape/Reshape_7*
_output_shapes
: *
dtype0*
valueB
 *   @k
gradient_tape/Mul_2Mul	sub_6:z:0gradient_tape/Const_7:output:0*
T0*
_output_shapes	
:�{
gradient_tape/Mul_3Mul gradient_tape/Reshape_7:output:0gradient_tape/Mul_2:z:0*
T0*
_output_shapes	
:�y
*gradient_tape/clip_by_value_1/zeros_like_1Const*
_output_shapes	
:�*
dtype0*
valueB�*    �
'gradient_tape/clip_by_value_1/LessEqual	LessEqual	Exp_1:y:0"clip_by_value_1/Minimum/y:output:0*
T0*
_output_shapes	
:��
(gradient_tape/clip_by_value_1/SelectV2_1SelectV2+gradient_tape/clip_by_value_1/LessEqual:z:0/gradient_tape/clip_by_value_1/SelectV2:output:03gradient_tape/clip_by_value_1/zeros_like_1:output:0*
T0*
_output_shapes	
:�n
gradient_tape/truediv_1/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      p
gradient_tape/truediv_1/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      �
-gradient_tape/truediv_1/BroadcastGradientArgsBroadcastGradientArgs&gradient_tape/truediv_1/Shape:output:0(gradient_tape/truediv_1/Shape_1:output:0*2
_output_shapes 
:���������:���������}
gradient_tape/truediv_1/RealDivRealDivgradient_tape/mul/Mul:z:0Sum:output:0*
T0*
_output_shapes
:	��
gradient_tape/truediv_1/SumSum#gradient_tape/truediv_1/RealDiv:z:02gradient_tape/truediv_1/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:	��
gradient_tape/truediv_1/ReshapeReshape$gradient_tape/truediv_1/Sum:output:0&gradient_tape/truediv_1/Shape:output:0*
T0*
_output_shapes
:	�U
gradient_tape/truediv_1/NegNegExp:y:0*
T0*
_output_shapes
:	��
!gradient_tape/truediv_1/RealDiv_1RealDivgradient_tape/truediv_1/Neg:y:0Sum:output:0*
T0*
_output_shapes
:	��
!gradient_tape/truediv_1/RealDiv_2RealDiv%gradient_tape/truediv_1/RealDiv_1:z:0Sum:output:0*
T0*
_output_shapes
:	��
gradient_tape/truediv_1/mulMulgradient_tape/mul/Mul:z:0%gradient_tape/truediv_1/RealDiv_2:z:0*
T0*
_output_shapes
:	��
gradient_tape/truediv_1/Sum_1Sumgradient_tape/truediv_1/mul:z:02gradient_tape/truediv_1/BroadcastGradientArgs:r1:0*
T0*
_output_shapes	
:��
!gradient_tape/truediv_1/Reshape_1Reshape&gradient_tape/truediv_1/Sum_1:output:0(gradient_tape/truediv_1/Shape_1:output:0*
T0*
_output_shapes
:	�}
,gradient_tape/sub_3/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"      }
,gradient_tape/sub_3/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB"      �
)gradient_tape/sub_3/BroadcastGradientArgsBroadcastGradientArgs5gradient_tape/sub_3/BroadcastGradientArgs/s0:output:05gradient_tape/sub_3/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:���������:���������s
)gradient_tape/sub_3/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
gradient_tape/sub_3/SumSumgradient_tape/mul/Mul_1:z:02gradient_tape/sub_3/Sum/reduction_indices:output:0*
T0*
_output_shapes	
:�r
!gradient_tape/sub_3/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/sub_3/ReshapeReshape gradient_tape/sub_3/Sum:output:0*gradient_tape/sub_3/Reshape/shape:output:0*
T0*
_output_shapes
:	�e
gradient_tape/sub_3/NegNeggradient_tape/mul/Mul_1:z:0*
T0*
_output_shapes
:	�]
gradient_tape/sub_5/NegNeggradient_tape/Mul_1:z:0*
T0*
_output_shapes	
:�]
gradient_tape/sub_6/NegNeggradient_tape/Mul_3:z:0*
T0*
_output_shapes	
:��
AddNAddNgradient_tape/mul_2/Mul_1:z:01gradient_tape/clip_by_value_1/SelectV2_1:output:0*
N*
T0*
_output_shapes	
:�W
gradient_tape/mul_4Mul
AddN:sum:0	Exp_1:y:0*
T0*
_output_shapes	
:�|
gradient_tape/Reciprocal
ReciprocalSum:output:0^gradient_tape/sub_3/Reshape*
T0*
_output_shapes
:	��
gradient_tape/mul_5Mul$gradient_tape/sub_3/Reshape:output:0gradient_tape/Reciprocal:y:0*
T0*
_output_shapes
:	�]
gradient_tape/sub_7/NegNeggradient_tape/mul_4:z:0*
T0*
_output_shapes	
:��
AddN_1AddN*gradient_tape/truediv_1/Reshape_1:output:0gradient_tape/mul_5:z:0*
N*
T0*
_output_shapes
:	�n
gradient_tape/Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/Reshape_8ReshapeAddN_1:sum:0&gradient_tape/Reshape_8/shape:output:0*
T0*
_output_shapes
:	�o
gradient_tape/Tile_4/multiplesConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/Tile_4Tile gradient_tape/Reshape_8:output:0'gradient_tape/Tile_4/multiples:output:0*
T0*
_output_shapes
:	�u
&gradient_tape/clip_by_value/zeros_likeConst*
_output_shapes	
:�*
dtype0*
valueB�*    �
(gradient_tape/clip_by_value/GreaterEqualGreaterEqualclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*
_output_shapes	
:��
$gradient_tape/clip_by_value/SelectV2SelectV2,gradient_tape/clip_by_value/GreaterEqual:z:0gradient_tape/Mul_3:z:0/gradient_tape/clip_by_value/zeros_like:output:0*
T0*
_output_shapes	
:��
5gradient_tape/softmax_cross_entropy_with_logits/ShapeConst*
_output_shapes
:*
dtype0*
valueB:��
7gradient_tape/softmax_cross_entropy_with_logits/ReshapeReshapegradient_tape/sub_7/Neg:y:0>gradient_tape/softmax_cross_entropy_with_logits/Shape:output:0*
T0*
_output_shapes	
:��
AddN_2AddN(gradient_tape/truediv_1/Reshape:output:0gradient_tape/Tile_4:output:0*
N*
T0*
_output_shapes
:	�[
gradient_tape/mul_6MulAddN_2:sum:0Exp:y:0*
T0*
_output_shapes
:	�w
(gradient_tape/clip_by_value/zeros_like_1Const*
_output_shapes	
:�*
dtype0*
valueB�*    �
%gradient_tape/clip_by_value/LessEqual	LessEqual	sub_4:z:0 clip_by_value/Minimum/y:output:0*
T0*
_output_shapes	
:��
&gradient_tape/clip_by_value/SelectV2_1SelectV2)gradient_tape/clip_by_value/LessEqual:z:0-gradient_tape/clip_by_value/SelectV2:output:01gradient_tape/clip_by_value/zeros_like_1:output:0*
T0*
_output_shapes	
:��
>gradient_tape/softmax_cross_entropy_with_logits/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
:gradient_tape/softmax_cross_entropy_with_logits/ExpandDims
ExpandDims@gradient_tape/softmax_cross_entropy_with_logits/Reshape:output:0Ggradient_tape/softmax_cross_entropy_with_logits/ExpandDims/dim:output:0*
T0*
_output_shapes
:	��
3gradient_tape/softmax_cross_entropy_with_logits/mulMulCgradient_tape/softmax_cross_entropy_with_logits/ExpandDims:output:0,softmax_cross_entropy_with_logits:backprop:0*
T0*
_output_shapes
:	��
:gradient_tape/softmax_cross_entropy_with_logits/LogSoftmax
LogSoftmax2softmax_cross_entropy_with_logits/Reshape:output:0*
T0*
_output_shapes
:	��
3gradient_tape/softmax_cross_entropy_with_logits/NegNegGgradient_tape/softmax_cross_entropy_with_logits/LogSoftmax:logsoftmax:0*
T0*
_output_shapes
:	��
@gradient_tape/softmax_cross_entropy_with_logits/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
<gradient_tape/softmax_cross_entropy_with_logits/ExpandDims_1
ExpandDims@gradient_tape/softmax_cross_entropy_with_logits/Reshape:output:0Igradient_tape/softmax_cross_entropy_with_logits/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:	��
5gradient_tape/softmax_cross_entropy_with_logits/mul_1MulEgradient_tape/softmax_cross_entropy_with_logits/ExpandDims_1:output:07gradient_tape/softmax_cross_entropy_with_logits/Neg:y:0*
T0*
_output_shapes
:	�w
AddN_3AddNgradient_tape/sub_3/Neg:y:0gradient_tape/mul_6:z:0*
N*
T0*
_output_shapes
:	�}
,gradient_tape/sub_2/BroadcastGradientArgs/s0Const*
_output_shapes
:*
dtype0*
valueB"      }
,gradient_tape/sub_2/BroadcastGradientArgs/s1Const*
_output_shapes
:*
dtype0*
valueB"      �
)gradient_tape/sub_2/BroadcastGradientArgsBroadcastGradientArgs5gradient_tape/sub_2/BroadcastGradientArgs/s0:output:05gradient_tape/sub_2/BroadcastGradientArgs/s1:output:0*2
_output_shapes 
:���������:���������V
gradient_tape/sub_2/NegNegAddN_3:sum:0*
T0*
_output_shapes
:	�s
)gradient_tape/sub_2/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:�
gradient_tape/sub_2/SumSumgradient_tape/sub_2/Neg:y:02gradient_tape/sub_2/Sum/reduction_indices:output:0*
T0*
_output_shapes	
:�r
!gradient_tape/sub_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      �
gradient_tape/sub_2/ReshapeReshape gradient_tape/sub_2/Sum:output:0*gradient_tape/sub_2/Reshape/shape:output:0*
T0*
_output_shapes
:	�u
gradient_tape/sub_4/NegNeg/gradient_tape/clip_by_value/SelectV2_1:output:0*
T0*
_output_shapes	
:��
7gradient_tape/softmax_cross_entropy_with_logits/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"      �
9gradient_tape/softmax_cross_entropy_with_logits/Reshape_1Reshape7gradient_tape/softmax_cross_entropy_with_logits/mul:z:0@gradient_tape/softmax_cross_entropy_with_logits/Shape_1:output:0*
T0*
_output_shapes
:	�f
gradient_tape/Shape_4Const*
_output_shapes
:*
dtype0*
valueB"      f
gradient_tape/Shape_5Const*
_output_shapes
:*
dtype0*
valueB"      l
gradient_tape/EqualEqualMax:output:0dense/BiasAdd:output:0*
T0*
_output_shapes
:	�l
gradient_tape/CastCastgradient_tape/Equal:z:0*

DstT0*

SrcT0
*
_output_shapes
:	�x
gradient_tape/Sum_4Sumgradient_tape/Cast:y:0Max/reduction_indices:output:0*
T0*
_output_shapes	
:��
gradient_tape/Reshape_9Reshapegradient_tape/Sum_4:output:0gradient_tape/Shape_5:output:0*
T0*
_output_shapes
:	��
gradient_tape/truediv_3RealDivgradient_tape/Cast:y:0 gradient_tape/Reshape_9:output:0*
T0*
_output_shapes
:	��
gradient_tape/mul_7Mulgradient_tape/truediv_3:z:0$gradient_tape/sub_2/Reshape:output:0*
T0*
_output_shapes
:	��
AddN_4AddNgradient_tape/Mul_1:z:0/gradient_tape/clip_by_value/SelectV2_1:output:0*
N*
T0*
_output_shapes	
:��
PartitionedCallPartitionedCallAddN_4:sum:0 StatefulPartitionedCall:output:1 StatefulPartitionedCall:output:2 StatefulPartitionedCall:output:3 StatefulPartitionedCall:output:4 StatefulPartitionedCall:output:5 StatefulPartitionedCall:output:6 StatefulPartitionedCall:output:7 StatefulPartitionedCall:output:8 StatefulPartitionedCall:output:9!StatefulPartitionedCall:output:10*
Tin
2*
Tout
2
*
_collective_manager_ids
 *d
_output_shapesR
P:@:@:@@:@:@@:@:@@:@:@:* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *1
f,R*
(__inference___backward_value_52003_52049�
AddN_5AddNAddN_3:sum:0Bgradient_tape/softmax_cross_entropy_with_logits/Reshape_1:output:0gradient_tape/mul_7:z:0*
N*
T0*
_output_shapes
:	�i
'gradient_tape/dense/BiasAdd/BiasAddGradBiasAddGradAddN_5:sum:0*
T0*
_output_shapes
:�
!gradient_tape/dense/MatMul/MatMulMatMulAddN_5:sum:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@*
transpose_b(�
#gradient_tape/dense/MatMul/MatMul_1MatMulmodel/mlp_fc3/Tanh:y:0AddN_5:sum:0*
T0*
_output_shapes

:@*
transpose_a(�
$gradient_tape/model/mlp_fc3/TanhGradTanhGradmodel/mlp_fc3/Tanh:y:0+gradient_tape/dense/MatMul/MatMul:product:0*
T0*
_output_shapes
:	�@�
/gradient_tape/model/mlp_fc3/BiasAdd/BiasAddGradBiasAddGrad(gradient_tape/model/mlp_fc3/TanhGrad:z:0*
T0*
_output_shapes
:@�
)gradient_tape/model/mlp_fc3/MatMul/MatMulMatMul(gradient_tape/model/mlp_fc3/TanhGrad:z:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@*
transpose_b(�
+gradient_tape/model/mlp_fc3/MatMul/MatMul_1MatMulmodel/mlp_fc2/Tanh:y:0(gradient_tape/model/mlp_fc3/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
$gradient_tape/model/mlp_fc2/TanhGradTanhGradmodel/mlp_fc2/Tanh:y:03gradient_tape/model/mlp_fc3/MatMul/MatMul:product:0*
T0*
_output_shapes
:	�@�
/gradient_tape/model/mlp_fc2/BiasAdd/BiasAddGradBiasAddGrad(gradient_tape/model/mlp_fc2/TanhGrad:z:0*
T0*
_output_shapes
:@�
)gradient_tape/model/mlp_fc2/MatMul/MatMulMatMul(gradient_tape/model/mlp_fc2/TanhGrad:z:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@*
transpose_b(�
+gradient_tape/model/mlp_fc2/MatMul/MatMul_1MatMulmodel/mlp_fc1/Tanh:y:0(gradient_tape/model/mlp_fc2/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
$gradient_tape/model/mlp_fc1/TanhGradTanhGradmodel/mlp_fc1/Tanh:y:03gradient_tape/model/mlp_fc2/MatMul/MatMul:product:0*
T0*
_output_shapes
:	�@�
/gradient_tape/model/mlp_fc1/BiasAdd/BiasAddGradBiasAddGrad(gradient_tape/model/mlp_fc1/TanhGrad:z:0*
T0*
_output_shapes
:@�
)gradient_tape/model/mlp_fc1/MatMul/MatMulMatMul(gradient_tape/model/mlp_fc1/TanhGrad:z:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@*
transpose_b(�
+gradient_tape/model/mlp_fc1/MatMul/MatMul_1MatMulmodel/mlp_fc0/Tanh:y:0(gradient_tape/model/mlp_fc1/TanhGrad:z:0*
T0*
_output_shapes

:@@*
transpose_a(�
$gradient_tape/model/mlp_fc0/TanhGradTanhGradmodel/mlp_fc0/Tanh:y:03gradient_tape/model/mlp_fc1/MatMul/MatMul:product:0*
T0*
_output_shapes
:	�@�
/gradient_tape/model/mlp_fc0/BiasAdd/BiasAddGradBiasAddGrad(gradient_tape/model/mlp_fc0/TanhGrad:z:0*
T0*
_output_shapes
:@�
)gradient_tape/model/mlp_fc0/MatMul/MatMulMatMulmodel/Cast:y:0(gradient_tape/model/mlp_fc0/TanhGrad:z:0*
T0*
_output_shapes

:@*
transpose_a(�
AddN_6AddNPartitionedCall:output:65gradient_tape/model/mlp_fc3/MatMul/MatMul_1:product:0*
N*
T0*
_output_shapes

:@@�
AddN_7AddNPartitionedCall:output:78gradient_tape/model/mlp_fc3/BiasAdd/BiasAddGrad:output:0*
N*
T0*
_output_shapes
:@�
AddN_8AddNPartitionedCall:output:03gradient_tape/model/mlp_fc0/MatMul/MatMul:product:0*
N*
T0*
_output_shapes

:@�
AddN_9AddNPartitionedCall:output:18gradient_tape/model/mlp_fc0/BiasAdd/BiasAddGrad:output:0*
N*
T0*
_output_shapes
:@�
AddN_10AddNPartitionedCall:output:25gradient_tape/model/mlp_fc1/MatMul/MatMul_1:product:0*
N*
T0*
_output_shapes

:@@�
AddN_11AddNPartitionedCall:output:38gradient_tape/model/mlp_fc1/BiasAdd/BiasAddGrad:output:0*
N*
T0*
_output_shapes
:@�
AddN_12AddNPartitionedCall:output:45gradient_tape/model/mlp_fc2/MatMul/MatMul_1:product:0*
N*
T0*
_output_shapes

:@@�
AddN_13AddNPartitionedCall:output:58gradient_tape/model/mlp_fc2/BiasAdd/BiasAddGrad:output:0*
N*
T0*
_output_shapes
:@�
global_norm/L2LossL2Loss-gradient_tape/dense/MatMul/MatMul_1:product:0*
T0*6
_class,
*(loc:@gradient_tape/dense/MatMul/MatMul_1*
_output_shapes
: �
global_norm/L2Loss_1L2Loss0gradient_tape/dense/BiasAdd/BiasAddGrad:output:0*
T0*:
_class0
.,loc:@gradient_tape/dense/BiasAdd/BiasAddGrad*
_output_shapes
: h
global_norm/L2Loss_2L2LossAddN_6:sum:0*
T0*
_class
loc:@AddN_6*
_output_shapes
: h
global_norm/L2Loss_3L2LossAddN_7:sum:0*
T0*
_class
loc:@AddN_7*
_output_shapes
: h
global_norm/L2Loss_4L2LossAddN_8:sum:0*
T0*
_class
loc:@AddN_8*
_output_shapes
: h
global_norm/L2Loss_5L2LossAddN_9:sum:0*
T0*
_class
loc:@AddN_9*
_output_shapes
: j
global_norm/L2Loss_6L2LossAddN_10:sum:0*
T0*
_class
loc:@AddN_10*
_output_shapes
: j
global_norm/L2Loss_7L2LossAddN_11:sum:0*
T0*
_class
loc:@AddN_11*
_output_shapes
: j
global_norm/L2Loss_8L2LossAddN_12:sum:0*
T0*
_class
loc:@AddN_12*
_output_shapes
: j
global_norm/L2Loss_9L2LossAddN_13:sum:0*
T0*
_class
loc:@AddN_13*
_output_shapes
: ~
global_norm/L2Loss_10L2LossPartitionedCall:output:8*
T0*"
_class
loc:@PartitionedCall*
_output_shapes
: ~
global_norm/L2Loss_11L2LossPartitionedCall:output:9*
T0*"
_class
loc:@PartitionedCall*
_output_shapes
: �
global_norm/stackPackglobal_norm/L2Loss:output:0global_norm/L2Loss_1:output:0global_norm/L2Loss_2:output:0global_norm/L2Loss_3:output:0global_norm/L2Loss_4:output:0global_norm/L2Loss_5:output:0global_norm/L2Loss_6:output:0global_norm/L2Loss_7:output:0global_norm/L2Loss_8:output:0global_norm/L2Loss_9:output:0global_norm/L2Loss_10:output:0global_norm/L2Loss_11:output:0*
N*
T0*
_output_shapes
:[
global_norm/ConstConst*
_output_shapes
:*
dtype0*
valueB: o
global_norm/SumSumglobal_norm/stack:output:0global_norm/Const:output:0*
T0*
_output_shapes
: X
global_norm/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *   @o
global_norm/mulMulglobal_norm/Sum:output:0global_norm/Const_1:output:0*
T0*
_output_shapes
: U
global_norm/global_normSqrtglobal_norm/mul:z:0*
T0*
_output_shapes
: b
clip_by_global_norm/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
clip_by_global_norm/truedivRealDiv&clip_by_global_norm/truediv/x:output:0global_norm/global_norm:y:0*
T0*
_output_shapes
: ^
clip_by_global_norm/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?d
clip_by_global_norm/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
clip_by_global_norm/truediv_1RealDiv"clip_by_global_norm/Const:output:0(clip_by_global_norm/truediv_1/y:output:0*
T0*
_output_shapes
: �
clip_by_global_norm/MinimumMinimumclip_by_global_norm/truediv:z:0!clip_by_global_norm/truediv_1:z:0*
T0*
_output_shapes
: ^
clip_by_global_norm/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?�
clip_by_global_norm/mulMul"clip_by_global_norm/mul/x:output:0clip_by_global_norm/Minimum:z:0*
T0*
_output_shapes
: y
clip_by_global_norm/subSubglobal_norm/global_norm:y:0global_norm/global_norm:y:0*
T0*
_output_shapes
: {
clip_by_global_norm/addAddV2clip_by_global_norm/mul:z:0clip_by_global_norm/sub:z:0*
T0*
_output_shapes
: �
clip_by_global_norm/mul_1Mul-gradient_tape/dense/MatMul/MatMul_1:product:0clip_by_global_norm/add:z:0*
T0*6
_class,
*(loc:@gradient_tape/dense/MatMul/MatMul_1*
_output_shapes

:@�
*clip_by_global_norm/clip_by_global_norm/_0Identityclip_by_global_norm/mul_1:z:0*
T0*6
_class,
*(loc:@gradient_tape/dense/MatMul/MatMul_1*
_output_shapes

:@�
clip_by_global_norm/mul_2Mul0gradient_tape/dense/BiasAdd/BiasAddGrad:output:0clip_by_global_norm/add:z:0*
T0*:
_class0
.,loc:@gradient_tape/dense/BiasAdd/BiasAddGrad*
_output_shapes
:�
*clip_by_global_norm/clip_by_global_norm/_1Identityclip_by_global_norm/mul_2:z:0*
T0*:
_class0
.,loc:@gradient_tape/dense/BiasAdd/BiasAddGrad*
_output_shapes
:�
clip_by_global_norm/mul_3MulAddN_6:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_6*
_output_shapes

:@@�
*clip_by_global_norm/clip_by_global_norm/_2Identityclip_by_global_norm/mul_3:z:0*
T0*
_class
loc:@AddN_6*
_output_shapes

:@@�
clip_by_global_norm/mul_4MulAddN_7:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_7*
_output_shapes
:@�
*clip_by_global_norm/clip_by_global_norm/_3Identityclip_by_global_norm/mul_4:z:0*
T0*
_class
loc:@AddN_7*
_output_shapes
:@�
clip_by_global_norm/mul_5MulAddN_8:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_8*
_output_shapes

:@�
*clip_by_global_norm/clip_by_global_norm/_4Identityclip_by_global_norm/mul_5:z:0*
T0*
_class
loc:@AddN_8*
_output_shapes

:@�
clip_by_global_norm/mul_6MulAddN_9:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_9*
_output_shapes
:@�
*clip_by_global_norm/clip_by_global_norm/_5Identityclip_by_global_norm/mul_6:z:0*
T0*
_class
loc:@AddN_9*
_output_shapes
:@�
clip_by_global_norm/mul_7MulAddN_10:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_10*
_output_shapes

:@@�
*clip_by_global_norm/clip_by_global_norm/_6Identityclip_by_global_norm/mul_7:z:0*
T0*
_class
loc:@AddN_10*
_output_shapes

:@@�
clip_by_global_norm/mul_8MulAddN_11:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_11*
_output_shapes
:@�
*clip_by_global_norm/clip_by_global_norm/_7Identityclip_by_global_norm/mul_8:z:0*
T0*
_class
loc:@AddN_11*
_output_shapes
:@�
clip_by_global_norm/mul_9MulAddN_12:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_12*
_output_shapes

:@@�
*clip_by_global_norm/clip_by_global_norm/_8Identityclip_by_global_norm/mul_9:z:0*
T0*
_class
loc:@AddN_12*
_output_shapes

:@@�
clip_by_global_norm/mul_10MulAddN_13:sum:0clip_by_global_norm/add:z:0*
T0*
_class
loc:@AddN_13*
_output_shapes
:@�
*clip_by_global_norm/clip_by_global_norm/_9Identityclip_by_global_norm/mul_10:z:0*
T0*
_class
loc:@AddN_13*
_output_shapes
:@�
clip_by_global_norm/mul_11MulPartitionedCall:output:8clip_by_global_norm/add:z:0*
T0*"
_class
loc:@PartitionedCall*
_output_shapes

:@�
+clip_by_global_norm/clip_by_global_norm/_10Identityclip_by_global_norm/mul_11:z:0*
T0*"
_class
loc:@PartitionedCall*
_output_shapes

:@�
clip_by_global_norm/mul_12MulPartitionedCall:output:9clip_by_global_norm/add:z:0*
T0*"
_class
loc:@PartitionedCall*
_output_shapes
:�
+clip_by_global_norm/clip_by_global_norm/_11Identityclip_by_global_norm/mul_12:z:0*
T0*"
_class
loc:@PartitionedCall*
_output_shapes
:y
IdentityIdentity3clip_by_global_norm/clip_by_global_norm/_0:output:0^NoOp*
T0*
_output_shapes

:@w

Identity_1Identity3clip_by_global_norm/clip_by_global_norm/_1:output:0^NoOp*
T0*
_output_shapes
:{

Identity_2Identity3clip_by_global_norm/clip_by_global_norm/_2:output:0^NoOp*
T0*
_output_shapes

:@@w

Identity_3Identity3clip_by_global_norm/clip_by_global_norm/_3:output:0^NoOp*
T0*
_output_shapes
:@{

Identity_4Identity3clip_by_global_norm/clip_by_global_norm/_4:output:0^NoOp*
T0*
_output_shapes

:@w

Identity_5Identity3clip_by_global_norm/clip_by_global_norm/_5:output:0^NoOp*
T0*
_output_shapes
:@{

Identity_6Identity3clip_by_global_norm/clip_by_global_norm/_6:output:0^NoOp*
T0*
_output_shapes

:@@w

Identity_7Identity3clip_by_global_norm/clip_by_global_norm/_7:output:0^NoOp*
T0*
_output_shapes
:@{

Identity_8Identity3clip_by_global_norm/clip_by_global_norm/_8:output:0^NoOp*
T0*
_output_shapes

:@@w

Identity_9Identity3clip_by_global_norm/clip_by_global_norm/_9:output:0^NoOp*
T0*
_output_shapes
:@}
Identity_10Identity4clip_by_global_norm/clip_by_global_norm/_10:output:0^NoOp*
T0*
_output_shapes

:@y
Identity_11Identity4clip_by_global_norm/clip_by_global_norm/_11:output:0^NoOp*
T0*
_output_shapes
:P
Identity_12IdentityMean_3:output:0^NoOp*
T0*
_output_shapes
: J
Identity_13Identity	mul_1:z:0^NoOp*
T0*
_output_shapes
: P
Identity_14IdentityMean_1:output:0^NoOp*
T0*
_output_shapes
: J
Identity_15Identity	mul_4:z:0^NoOp*
T0*
_output_shapes
: P
Identity_16IdentityMean_5:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^StatefulPartitionedCall^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"#
identity_10Identity_10:output:0"#
identity_11Identity_11:output:0"#
identity_12Identity_12:output:0"#
identity_13Identity_13:output:0"#
identity_14Identity_14:output:0"#
identity_15Identity_15:output:0"#
identity_16Identity_16:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"!

identity_9Identity_9:output:0*(
_construction_contextkEagerRuntime*Y
_input_shapesH
F:	�:�:�:�:�:�: : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:D @

_output_shapes
:	�

_user_specified_nameobs:D@

_output_shapes	
:�
!
_user_specified_name	returns:B>

_output_shapes	
:�

_user_specified_namemasks:D@

_output_shapes	
:�
!
_user_specified_name	actions:C?

_output_shapes	
:�
 
_user_specified_namevalues:JF

_output_shapes	
:�
'
_user_specified_nameneglogpac_old
�6
�

__forward_value_52048
observation	>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity!
dense_1_matmul_readvariableop
model_mlp_fc3_tanh'
#model_mlp_fc3_matmul_readvariableop
model_mlp_fc2_tanh'
#model_mlp_fc2_matmul_readvariableop
model_mlp_fc1_tanh'
#model_mlp_fc1_matmul_readvariableop
model_mlp_fc0_tanh'
#model_mlp_fc0_matmul_readvariableop

model_cast��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOpX

model/CastCastobservation*

DstT0*

SrcT0	*
_output_shapes
:	��
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@d
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes
:	�@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMulmodel/mlp_fc3/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	��
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	�i
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
S
IdentityIdentitySqueeze:output:0^NoOp*
T0*
_output_shapes	
:��
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "F
dense_1_matmul_readvariableop%dense_1/MatMul/ReadVariableOp:value:0"
identityIdentity:output:0"

model_castmodel/Cast:y:0"R
#model_mlp_fc0_matmul_readvariableop+model/mlp_fc0/MatMul/ReadVariableOp:value:0",
model_mlp_fc0_tanhmodel/mlp_fc0/Tanh:y:0"R
#model_mlp_fc1_matmul_readvariableop+model/mlp_fc1/MatMul/ReadVariableOp:value:0",
model_mlp_fc1_tanhmodel/mlp_fc1/Tanh:y:0"R
#model_mlp_fc2_matmul_readvariableop+model/mlp_fc2/MatMul/ReadVariableOp:value:0",
model_mlp_fc2_tanhmodel/mlp_fc2/Tanh:y:0"R
#model_mlp_fc3_matmul_readvariableop+model/mlp_fc3/MatMul/ReadVariableOp:value:0",
model_mlp_fc3_tanhmodel/mlp_fc3/Tanh:y:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:	�: : : : : : : : : : *D
backward_function_name*(__inference___backward_value_52003_520492@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:L H

_output_shapes
:	�
%
_user_specified_nameobservation
�.
�
__inference_value_52319
observation	>
,model_mlp_fc0_matmul_readvariableop_resource:@;
-model_mlp_fc0_biasadd_readvariableop_resource:@>
,model_mlp_fc1_matmul_readvariableop_resource:@@;
-model_mlp_fc1_biasadd_readvariableop_resource:@>
,model_mlp_fc2_matmul_readvariableop_resource:@@;
-model_mlp_fc2_biasadd_readvariableop_resource:@>
,model_mlp_fc3_matmul_readvariableop_resource:@@;
-model_mlp_fc3_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity��dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�$model/mlp_fc0/BiasAdd/ReadVariableOp�#model/mlp_fc0/MatMul/ReadVariableOp�$model/mlp_fc1/BiasAdd/ReadVariableOp�#model/mlp_fc1/MatMul/ReadVariableOp�$model/mlp_fc2/BiasAdd/ReadVariableOp�#model/mlp_fc2/MatMul/ReadVariableOp�$model/mlp_fc3/BiasAdd/ReadVariableOp�#model/mlp_fc3/MatMul/ReadVariableOpW

model/CastCastobservation*

DstT0*

SrcT0	*
_output_shapes

:�
#model/mlp_fc0/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc0_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/mlp_fc0/MatMulMatMulmodel/Cast:y:0+model/mlp_fc0/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc0/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc0_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc0/BiasAddBiasAddmodel/mlp_fc0/MatMul:product:0,model/mlp_fc0/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc0/TanhTanhmodel/mlp_fc0/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc1/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc1/MatMulMatMulmodel/mlp_fc0/Tanh:y:0+model/mlp_fc1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc1/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc1/BiasAddBiasAddmodel/mlp_fc1/MatMul:product:0,model/mlp_fc1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc1/TanhTanhmodel/mlp_fc1/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc2/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc2/MatMulMatMulmodel/mlp_fc1/Tanh:y:0+model/mlp_fc2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc2/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc2/BiasAddBiasAddmodel/mlp_fc2/MatMul:product:0,model/mlp_fc2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc2/TanhTanhmodel/mlp_fc2/BiasAdd:output:0*
T0*
_output_shapes

:@�
#model/mlp_fc3/MatMul/ReadVariableOpReadVariableOp,model_mlp_fc3_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/mlp_fc3/MatMulMatMulmodel/mlp_fc2/Tanh:y:0+model/mlp_fc3/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:@�
$model/mlp_fc3/BiasAdd/ReadVariableOpReadVariableOp-model_mlp_fc3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/mlp_fc3/BiasAddBiasAddmodel/mlp_fc3/MatMul:product:0,model/mlp_fc3/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:@c
model/mlp_fc3/TanhTanhmodel/mlp_fc3/BiasAdd:output:0*
T0*
_output_shapes

:@�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense_1/MatMulMatMulmodel/mlp_fc3/Tanh:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:h
SqueezeSqueezedense_1/BiasAdd:output:0*
T0*
_output_shapes
:*
squeeze_dims
R
IdentityIdentitySqueeze:output:0^NoOp*
T0*
_output_shapes
:�
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp%^model/mlp_fc0/BiasAdd/ReadVariableOp$^model/mlp_fc0/MatMul/ReadVariableOp%^model/mlp_fc1/BiasAdd/ReadVariableOp$^model/mlp_fc1/MatMul/ReadVariableOp%^model/mlp_fc2/BiasAdd/ReadVariableOp$^model/mlp_fc2/MatMul/ReadVariableOp%^model/mlp_fc3/BiasAdd/ReadVariableOp$^model/mlp_fc3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:: : : : : : : : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2L
$model/mlp_fc0/BiasAdd/ReadVariableOp$model/mlp_fc0/BiasAdd/ReadVariableOp2J
#model/mlp_fc0/MatMul/ReadVariableOp#model/mlp_fc0/MatMul/ReadVariableOp2L
$model/mlp_fc1/BiasAdd/ReadVariableOp$model/mlp_fc1/BiasAdd/ReadVariableOp2J
#model/mlp_fc1/MatMul/ReadVariableOp#model/mlp_fc1/MatMul/ReadVariableOp2L
$model/mlp_fc2/BiasAdd/ReadVariableOp$model/mlp_fc2/BiasAdd/ReadVariableOp2J
#model/mlp_fc2/MatMul/ReadVariableOp#model/mlp_fc2/MatMul/ReadVariableOp2L
$model/mlp_fc3/BiasAdd/ReadVariableOp$model/mlp_fc3/BiasAdd/ReadVariableOp2J
#model/mlp_fc3/MatMul/ReadVariableOp#model/mlp_fc3/MatMul/ReadVariableOp:K G

_output_shapes

:
%
_user_specified_nameobservation
�

�
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861763

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������@W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:��
�
train_model
	optimizer

loss_names
get_grad
step
	value

signatures"
_generic_user_object
t
policy_network
value_network

	pdtype

value_fc
step
	value"
_generic_user_object
�
iter

beta_1

beta_2
	decay
learning_rate'm�(m�/m�0m�7m�8m�?m�@m�Gm�Hm�\m�]m�'v�(v�/v�0v�7v�8v�?v�@v�Gv�Hv�\v�]v�"
	optimizer
 "
trackable_list_wrapper
�
trace_02�
__inference_get_grad_52166�
���
FullArgSpeca
argsY�V
jself
j	cliprange
jobs
	jreturns
jmasks
	jactions
jvalues
jneglogpac_old
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_02�
__inference_step_52279�
���
FullArgSpec"
args�
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0
�
trace_0
trace_12�
__inference_value_52319
__inference_value_52359�
���
FullArgSpec"
args�
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
"
signature_map
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
/
 matching_fc"
_generic_user_object
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
�B�
__inference_get_grad_52166obsreturnsmasksactionsvaluesneglogpac_old"�
���
FullArgSpeca
argsY�V
jself
j	cliprange
jobs
	jreturns
jmasks
	jactions
jvalues
jneglogpac_old
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_step_52279observation"�
���
FullArgSpec"
args�
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_value_52319observation"�
���
FullArgSpec"
args�
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_value_52359observation"�
���
FullArgSpec"
args�
jself
jobservation
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_tf_keras_input_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses

7kernel
8bias"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

?kernel
@bias"
_tf_keras_layer
�
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses

Gkernel
Hbias"
_tf_keras_layer
X
/0
01
72
83
?4
@5
G6
H7"
trackable_list_wrapper
X
/0
01
72
83
?4
@5
G6
H7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
Ntrace_0
Otrace_1
Ptrace_2
Qtrace_32�
'__inference_model_layer_call_fn_1861462
'__inference_model_layer_call_fn_1861658
'__inference_model_layer_call_fn_1861679
'__inference_model_layer_call_fn_1861589�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zNtrace_0zOtrace_1zPtrace_2zQtrace_3
�
Rtrace_0
Strace_1
Ttrace_2
Utrace_32�
B__inference_model_layer_call_and_return_conditional_losses_1861711
B__inference_model_layer_call_and_return_conditional_losses_1861743
B__inference_model_layer_call_and_return_conditional_losses_1861613
B__inference_model_layer_call_and_return_conditional_losses_1861637�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0zStrace_1zTtrace_2zUtrace_3
�B�
"__inference__wrapped_model_1861367input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses

\kernel
]bias"
_tf_keras_layer
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:@2	vf/kernel
:2vf/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cnon_trainable_variables

dlayers
emetrics
flayer_regularization_losses
glayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
htrace_02�
)__inference_mlp_fc0_layer_call_fn_1861752�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zhtrace_0
�
itrace_02�
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861763�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zitrace_0
 :@2mlp_fc0/kernel
:@2mlp_fc0/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
jnon_trainable_variables

klayers
lmetrics
mlayer_regularization_losses
nlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
otrace_02�
)__inference_mlp_fc1_layer_call_fn_1861772�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zotrace_0
�
ptrace_02�
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861783�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zptrace_0
 :@@2mlp_fc1/kernel
:@2mlp_fc1/bias
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
qnon_trainable_variables

rlayers
smetrics
tlayer_regularization_losses
ulayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
vtrace_02�
)__inference_mlp_fc2_layer_call_fn_1861792�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zvtrace_0
�
wtrace_02�
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861803�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zwtrace_0
 :@@2mlp_fc2/kernel
:@2mlp_fc2/bias
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
�
}trace_02�
)__inference_mlp_fc3_layer_call_fn_1861812�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z}trace_0
�
~trace_02�
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861823�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z~trace_0
 :@@2mlp_fc3/kernel
:@2mlp_fc3/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_model_layer_call_fn_1861462input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1861658inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1861679inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_model_layer_call_fn_1861589input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1861711inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1861743inputs"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1861613input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_model_layer_call_and_return_conditional_losses_1861637input_1"�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
\0
]1"
trackable_list_wrapper
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:@2	pi/kernel
:2pi/bias
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
�B�
)__inference_mlp_fc0_layer_call_fn_1861752inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861763inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_mlp_fc1_layer_call_fn_1861772inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861783inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_mlp_fc2_layer_call_fn_1861792inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861803inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
�B�
)__inference_mlp_fc3_layer_call_fn_1861812inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861823inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
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
 :@2Adam/vf/kernel/m
:2Adam/vf/bias/m
%:#@2Adam/mlp_fc0/kernel/m
:@2Adam/mlp_fc0/bias/m
%:#@@2Adam/mlp_fc1/kernel/m
:@2Adam/mlp_fc1/bias/m
%:#@@2Adam/mlp_fc2/kernel/m
:@2Adam/mlp_fc2/bias/m
%:#@@2Adam/mlp_fc3/kernel/m
:@2Adam/mlp_fc3/bias/m
 :@2Adam/pi/kernel/m
:2Adam/pi/bias/m
 :@2Adam/vf/kernel/v
:2Adam/vf/bias/v
%:#@2Adam/mlp_fc0/kernel/v
:@2Adam/mlp_fc0/bias/v
%:#@@2Adam/mlp_fc1/kernel/v
:@2Adam/mlp_fc1/bias/v
%:#@@2Adam/mlp_fc2/kernel/v
:@2Adam/mlp_fc2/bias/v
%:#@@2Adam/mlp_fc3/kernel/v
:@2Adam/mlp_fc3/bias/v
 :@2Adam/pi/kernel/v
:2Adam/pi/bias/v�
"__inference__wrapped_model_1861367o/078?@GH0�-
&�#
!�
input_1���������
� "1�.
,
mlp_fc3!�
mlp_fc3���������@�
__inference_get_grad_52166�/078?@GH\]'(���
���
	Y�������?
�
obs	�	
�
returns�
�
masks�

�
actions�	
�
values�
�
neglogpac_old�
� "���
���
�
0/0@
�
0/1
�
0/2@@
�
0/3@
�
0/4@
�
0/5@
�
0/6@@
�
0/7@
�
0/8@@
�
0/9@
�
0/10@
�
0/11

�
1 

�
2 

�
3 

�
4 

�
5 �
D__inference_mlp_fc0_layer_call_and_return_conditional_losses_1861763\/0/�,
%�"
 �
inputs���������
� "%�"
�
0���������@
� |
)__inference_mlp_fc0_layer_call_fn_1861752O/0/�,
%�"
 �
inputs���������
� "����������@�
D__inference_mlp_fc1_layer_call_and_return_conditional_losses_1861783\78/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_mlp_fc1_layer_call_fn_1861772O78/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_mlp_fc2_layer_call_and_return_conditional_losses_1861803\?@/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_mlp_fc2_layer_call_fn_1861792O?@/�,
%�"
 �
inputs���������@
� "����������@�
D__inference_mlp_fc3_layer_call_and_return_conditional_losses_1861823\GH/�,
%�"
 �
inputs���������@
� "%�"
�
0���������@
� |
)__inference_mlp_fc3_layer_call_fn_1861812OGH/�,
%�"
 �
inputs���������@
� "����������@�
B__inference_model_layer_call_and_return_conditional_losses_1861613k/078?@GH8�5
.�+
!�
input_1���������
p 

 
� "%�"
�
0���������@
� �
B__inference_model_layer_call_and_return_conditional_losses_1861637k/078?@GH8�5
.�+
!�
input_1���������
p

 
� "%�"
�
0���������@
� �
B__inference_model_layer_call_and_return_conditional_losses_1861711j/078?@GH7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������@
� �
B__inference_model_layer_call_and_return_conditional_losses_1861743j/078?@GH7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������@
� �
'__inference_model_layer_call_fn_1861462^/078?@GH8�5
.�+
!�
input_1���������
p 

 
� "����������@�
'__inference_model_layer_call_fn_1861589^/078?@GH8�5
.�+
!�
input_1���������
p

 
� "����������@�
'__inference_model_layer_call_fn_1861658]/078?@GH7�4
-�*
 �
inputs���������
p 

 
� "����������@�
'__inference_model_layer_call_fn_1861679]/078?@GH7�4
-�*
 �
inputs���������
p

 
� "����������@�
__inference_step_52279t/078?@GH\]'(+�(
!�
�
observation	
� "7�4
�
0	
�
1

 
�
3a
__inference_value_52319F
/078?@GH'(+�(
!�
�
observation	
� "�c
__inference_value_52359H
/078?@GH'(,�)
"�
�
observation	�	
� "�	�