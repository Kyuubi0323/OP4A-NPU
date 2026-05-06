# ACUITY Toolkit Usage Example

Using the MobileNetV2 object recognition model in Keras format as an example, the model will be parsed, quantized, and compiled using the Acuity Toolkit, and project code will be generated, and the Vivante IDE will be used for simulation, and finally inference will be performed on the VIP9000 series NPUs.

## NPU Version Comparison Table

| Products  | Hashrate | Platform | NPU Version | NPU Software Version |
|-----------|----------|----------|-------------|----------------------|
| Radxa A5E | 2 Tops   | T527     | v2          | v1.13                |
| Radxa A7A | 3 Tops   | A733     | v3          | v2.0                 |
| Radxa A7Z | 3 Tops   | A733     | v3          | v2.0                 |
| Radxa A7S | 3 Tops   | A733     | v3          | v2.0                 |

## Download the Example Repository

Download the sample repository in the ACUITY Docker container.

**X86 Linux PC**
```shell
git clone https://github.com/ZIFENG278/ai-sdk.git
```

## Configure the Model Compilation Script

**Docker**
```shell
cd ai-sdk/models
source env.sh v2  # NPU_VERSION
cp ../scripts/* .
```

> **Tips:** Specify `NPU_VERSION` — `v3` for A733, `v2` for T527. For more information, refer to the [NPU version comparison table](#npu-version-comparison-table).

## Model Analysis

Go to the `ai-sdk/models/MobileNetV2_Imagenet` model example directory in the sample repository in Docker.

**Docker**
```shell
cd ai-sdk/models/MobileNetV2_Imagenet
```

The following files are included in this directory:

- `MobileNetV2_Imagenet.h5` — the original model file (required).
- `channel_mean_value.txt` — a file of the mean and scale values entered for the model.
- `dataset.txt` — calibration file set files quantified for the model.
- `space_shuttle_224x224.jpg` — input images for the test and calibration images included in the standard set.
- `inputs_outputs.txt` — contains the model input and output node files. (If necessary, you need to set up output nodes to avoid quantization failures.)

```
.
|-- MobileNetV2_Imagenet.h5
|-- channel_mean_value.txt
|-- dataset.txt
|-- inputs_outputs.txt
`-- space_shuttle_224x224.jpg

0 directories, 5 files
```

## Import the Model

`pegasus_import.sh` — the model import script can parse the model structure and weights of a variety of different AI framework models, and output the parsed files:

- The model schema is saved in `MODEL_DIR.json`
- Model weights are saved in `MODEL_DIR.data`
- Automatically generate model input file templates `MODEL_DIR_inputmeta.yml`
- Automatically generate model post-processing file templates `MODEL_DIR_postprocess_file.yml`

**Docker**
```shell
# pegasus_import.sh MODEL_DIR :D remember to put the name of the folder same as the model name !!!
./pegasus_import.sh MobileNetV2_Imagenet/
```

**Parameters:**

- `MODEL_DIR` — the folder containing the source model files

## Manually Modify the Model Input File

Here you need to manually set the mean and scale in `MobileNetV2_Imagenet_inputmeta.yml` according to the preprocessed mean and scale according to the model input. Here is an example of MobileNetV2_ImageNet — because the output of MobileNet is `(1,224,224,3)` RGB tri-channel, according to the model preprocessing formula:

```
x1 = (x - mean) / std
x1 = (x - mean) * scale
scale = 1 / std
```

Since the training dataset is ImageNet, the normalized mean of the ImageNet training set here is `[0.485, 0.456, 0.406]`, and the std is `[0.229, 0.224, 0.225]`. Here it is necessary to perform denormalization calculations. Normalized data refer to the pytorch documentation.

```
# mean
0.485 * 255 = 123.675
0.456 * 255 = 116.28
0.406 * 255 = 103.53

# scale
1 / (0.229 * 255) = 0.01712
1 / (0.224 * 255) = 0.01751
1 / (0.225 * 255) = 0.01743
```

Here is the calculation of the values of mean and scale to set in `MobileNetV2_Imagenet_inputmeta.yml`:

```yaml
mean:
- 123.675
- 116.28
- 103.53
scale:
- 0.01712
- 0.01751
- 0.01743
```

## Quantitative Models

Before the model is converted, the model can be quantized in different types. ACUITY supports `uint8` / `int16` / `bf16` / `pcq` (int8 per-channel quantized) quantization methods. If `float` is used, it means that no quantization is performed.

Use `pegasus_quantize.sh` to quantify a specified type of model.

> **Tips:** If the source model itself is already a quantization model, there is no need to quantize it here, otherwise an error will be reported.

**Docker**
```shell
# pegasus_quantize.sh MODEL_DIR QUANTIZED ITERATION
pegasus_quantize.sh MobileNetV2_Imagenet int16 10
```

Quantization generates a quantization file `MODEL_DIR_QUANTIZED.quantize`.

| QUANTIZED | TYPE  | QUANTIZER                    |
|-----------|-------|------------------------------|
| uint8     | uint8 | asymmetric_affine            |
| int16     | int16 | dynamic_fixed_point          |
| pcq       | int8  | perchannel_symmetric_affine  |
| bf16      | bf16  | qbfloat16                    |

## Inference Quantitative Model

After the model is quantized, the performance will be improved to varying degrees, but the accuracy will be slightly reduced. The input for the test inference (`space_shuttle`) is the first image in the `dataset.txt`.

### Inference Float Type Model

The unquantized float model is inferred, and the results are obtained as a reference for the quantified model results.

**Docker**
```shell
# pegasus_inference.sh MODEL_DIR QUANTIZED ITERATION
pegasus_inference.sh MobileNetV2_Imagenet/ float
```

The inference result is output as:

```
I 07:01:06 Iter(0), top(5), tensor(@attach_Logits/Softmax/out0_0:out0) :
I 07:01:06 812: 0.9990391731262207
I 07:01:06 814: 0.0001562383840791881
I 07:01:06 627: 8.89502334757708e-05
I 07:01:06 864: 6.59249781165272e-05
I 07:01:06 536: 2.808812860166654e-05
```

The highest confidence level of the top 5 output here is `812`, and the corresponding label is `space shuttle`, which is consistent with the actual input image type, which indicates that the mean and scale settings of the model's input preprocessing are correct.

The tensor of inference is also saved in the `MODEL_DIR/inf/MODEL_DIR_QUANTIZED` folder:

- `iter_0_input_1_158_out0_1_224_224_3.qnt.tensor` — the tensor of the original image
- `iter_0_input_1_158_out0_1_224_224_3.tensor` — inputs tensor to the preprocessed model
- `iter_0_attach_Logits_Softmax_out0_0_out0_1_1000.tensor` — the output tensor of the model

```
.
|-- iter_0_attach_Logits_Softmax_out0_0_out0_1_1000.tensor
|-- iter_0_input_1_158_out0_1_224_224_3.qnt.tensor
`-- iter_0_input_1_158_out0_1_224_224_3.tensor

0 directories, 3 files
```

### Inference uint8 Quantization Model

**Docker**
```shell
# pegasus_inference.sh MODEL_DIR QUANTIZED ITERATION
pegasus_inference.sh MobileNetV2_Imagenet/ uint8
```

The inference result is output as:

```
I 07:02:20 Iter(0), top(5), tensor(@attach_Logits/Softmax/out0_0:out0) :
I 07:02:20 904: 0.8729746341705322
I 07:02:20 530: 0.012925799004733562
I 07:02:20 905: 0.01022859662771225
I 07:02:20 468: 0.006405209191143513
I 07:02:20 466: 0.005068646278232336
```

> **Note:** The highest confidence level of the top 5 output here is `904`. The label corresponding to `904` is not consistent with the input image result, and it is inconsistent with the inference results of the float type — this means that there is a loss of precision after uint8 quantization. A higher precision quantization model such as `pcq` or `int16` can be applied. For methods to improve model accuracy, please refer to Mixed Quantization.

### Inference PCQ Quantitative Models

**Docker**
```shell
# pegasus_inference.sh MODEL_DIR QUANTIZED ITERATION
pegasus_inference.sh MobileNetV2_Imagenet/ pcq
```

The inference result is output as:

```
I 03:36:41 Iter(0), top(5), tensor(@attach_Logits/Softmax/out0_0:out0) :
I 03:36:41 812: 0.9973124265670776
I 03:36:41 814: 0.00034916045842692256
I 03:36:41 627: 0.00010834729619091377
I 03:36:41 833: 9.26952125155367e-05
I 03:36:41 576: 6.784773722756654e-05
```

The highest confidence level of the top 5 output here is `812`, and the corresponding label is `space shuttle`, which is consistent with the actual input image type and is consistent with the float type inference results, which indicates that the accuracy is correct when quantizing PCQ.

### Inference int16 Quantization Model

**Docker**
```shell
# pegasus_inference.sh MODEL_DIR QUANTIZED ITERATION
pegasus_inference.sh MobileNetV2_Imagenet/ int16
```

The inference result is output as:

```
I 06:54:23 Iter(0), top(5), tensor(@attach_Logits/Softmax/out0_0:out0) :
I 06:54:23 812: 0.9989829659461975
I 06:54:23 814: 0.0001675251842243597
I 06:54:23 627: 9.466391202295199e-05
I 06:54:23 864: 6.788487371522933e-05
I 06:54:23 536: 3.0241633794503286e-05
```

The highest confidence level of the top5 output here is `812`, and the corresponding label is `space shuttle`, which is consistent with the actual input image type and is consistent with the float type inference results, which indicates that the accuracy is correct when quantized by int16.

## The Model Is Compiled and Exported

`pegasus_export_ovx.sh` can export the model files and project code required for NPU inference.

Here is an example of an INT16 quantization model:

**Docker**
```shell
# pegasus_export_ovx.sh MODEL_DIR QUANTIZED
pegasus_export_ovx.sh MobileNetV2_Imagenet int16
```

Generate OpenVX project and NBG project path:

- `MODEL_DIR/wksp/MODEL_DIR_QUANTIZED` — a cross-platform OpenVX project that requires hardware just-in-time compilation (JIT) for model initialization.
- `MODEL_DIR/wksp/MODEL_DIR_QUANTIZED_nbg_unify` — NBG format, precompiled machine code format, low overhead, fast initialization.

```
(.venv) root@focal-v4l2:~/work/Acuity/acuity_examples/models/MobileNetV2_Imagenet/wksp$ ls
MobileNetV2_Imagenet_int16  MobileNetV2_Imagenet_int16_nbg_unify
```

> **Tips:** The model file `network_binary.nb` is available in the NBG project. The compiled model can be copied to the board side and use the `vpm_run` or `awnn` API for board-side inference.

## Run Inference Using Vivante IDE Simulations

The Vivante IDE allows you to validate the generated target model and OpenVX project in ACUITY Docker on an X86 PC.

### Import the Environment Variables Required for the Vivante IDE

**Docker**
```shell
export USE_IDE_LIB=1
export VIVANTE_SDK_DIR=~/Vivante_IDE/VivanteIDE5.8.2/cmdtools/vsimulator
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/Vivante_IDE/VivanteIDE5.8.2/cmdtools/common/lib:~/Vivante_IDE/VivanteIDE5.8.2/cmdtools/vsimulator/lib
unset VSI_USE_IMAGE_PROCESS
```

### Simulate Running a Cross-Platform OpenVX Project

**Compile the executable**

**Docker**
```shell
cd MobileNetV2_Imagenet/wksp/MobileNetV2_Imagenet_int16
make -f makefile.linux
```

The resulting target file is a binary executable of `MODEL_DIR_QUANTIZED`.

**Run the executable**

**Docker**
```shell
# Usage: ./mobilenetv2imagenetint16 data_file inputs...
./mobilenetv2imagenetint16 MobileNetV2_Imagenet_int16.export.data ../../space_shuttle_224x224.jpg
```

Run result:

```
Create Neural Network: 11ms or 11426us
Verify...
Verify Graph: 2430ms or 2430049us
Start run graph [1] times...
Run the 1 time: 229309.52ms or 229309520.00us
vxProcessGraph execution time:
Total   229309.53ms or 229309536.00us
Average 229309.53ms or 229309536.00us
 --- Top5 ---
812: 0.999023
814: 0.000146
627: 0.000084
864: 0.000067
  0: 0.000000
```

### Simulate Running an NBG Project

> **Tips:** Running an NBG project with the Vivante IDE takes a lot of time.

**Compile the executable**

**Docker**
```shell
cd MobileNetV2_Imagenet/wksp/MobileNetV2_Imagenet_int16_nbg_unify
make -f makefile.linux
```

The resulting target file is a binary executable of `MODEL_DIR_QUANTIZED`.

**Run the executable**

**Docker**
```shell
# Usage: ./mobilenetv2imagenetint16 data_file inputs...
./mobilenetv2imagenetint16 network_binary.nb ../../space_shuttle_224x224.jpg
```

Run result:

```
Create Neural Network: 4ms or 4368us
Verify...
Verify Graph: 2ms or 2482us
Start run graph [1] times...
Run the 1 time: 229388.50ms or 229388496.00us
vxProcessGraph execution time:
Total   229388.52ms or 229388512.00us
Average 229388.52ms or 229388512.00us
 --- Top5 ---
812: 0.999023
814: 0.000146
627: 0.000084
864: 0.000067
  0: 0.000000
```

## Board-Side NPU Inference

The board side uses the NPU to inference the NBG format model, which can be used for inference testing with `vpm_run`.

For `vpm_run` installation and use, please refer to the **vpm_run Model Test Tool**.
