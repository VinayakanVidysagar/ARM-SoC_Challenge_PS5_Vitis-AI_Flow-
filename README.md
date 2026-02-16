# Vitis AI

AMD Vitis™ AI is an integrated development environment that can be leveraged to accelerate AI inference on AMD platforms. This toolchain provides optimized IP, tools, libraries, models, as well as resources, such as example designs and tutorials that aid the user throughout the development process. It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on AMD Adaptable SoCs and Alveo Data Center accelerator cards.


The Vitis™ AI solution consists of three primary components:
    • The Deep-Learning Processor unit (DPU), a hardware engine for optimized the inferencing of ML models
    • Model development tools, to compile and optimize ML models for the DPU
    • Model deployment libraries and APIs, to integrate and execute the ML models on the DPU engine from a SW application
# About ZCU104 MP-SoC Evaluation Board

The ZCU104 Evaluation Board is built around the Xilinx Zynq UltraScale+ MPSoC, specifically the XCZU7EV device, which integrates a heterogeneous processing system with programmable logic on a single silicon die. This architecture combines a quad-core ARM Cortex-A53 application processing unit, a dual-core ARM Cortex-R5 real-time processing unit, and a Mali-400 MP2 GPU within the Processing System (PS), tightly coupled with high-performance FPGA fabric in the Programmable Logic (PL). This heterogeneous integration enables hardware–software co-design, allowing computationally intensive tasks such as deep learning acceleration to be offloaded to the PL while system control and application software execute on the ARM cores.

The ZCU104 provides substantial programmable logic resources, including hundreds of thousands of logic cells, DSP slices optimized for multiply–accumulate operations, block RAM (BRAM), and UltraRAM, making it highly suitable for implementing accelerators such as the Deep Learning Processing Unit (DPU). The presence of dedicated DSP48E2 slices enhances performance for convolution-heavy workloads, while on-chip memory resources reduce external memory bandwidth dependency and improve latency.

In terms of memory architecture, the board supports DDR4 memory connected to both the Processing System and Programmable Logic, enabling high-bandwidth data movement for AI inference, video processing, and signal processing applications. The MPSoC architecture also includes high-performance AXI interfaces that facilitate low-latency communication between PS and PL domains, which is critical for deploying Vitis AI–based acceleration pipelines.

The ZCU104 features rich high-speed connectivity options, including PCIe, DisplayPort, USB 3.0, Gigabit Ethernet, FMC connectors, and multiple MIPI interfaces. These interfaces enable integration with cameras, sensors, storage devices, and external accelerators. The board also includes SD card boot support, UART for serial debugging, and JTAG for hardware development and debugging, making it suitable for both prototyping and production-level validation.

From a power and performance standpoint, the Zynq UltraScale+ MPSoC uses a 16nm FinFET process technology, offering improved energy efficiency compared to earlier FPGA generations. This makes the ZCU104 particularly attractive for edge AI applications where performance-per-watt is a critical constraint. The integrated security features, including secure boot, hardware root of trust, and cryptographic acceleration, make it suitable for secure embedded deployments.

Overall, the ZCU104 MPSoC platform provides a balanced combination of general-purpose processing, real-time control capability, GPU support, and highly parallel programmable logic resources. This makes it an ideal development platform for applications such as real-time object detection, instance segmentation, embedded vision, robotics, industrial automation, and AI acceleration using Vitis AI and DPU-based architectures.

**Deep-Learning Processor Unit**

The Deep-learning Processor Unit (DPU) is a programmable engine optimized for deep neural networks. The DPU implements an efficient tensor-level instruction set designed to support and accelerate various popular convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO, SSD, and MobileNet, among others.
The DPU supports on AMD Zynq™ UltraScale+™ MPSoCs, the Kria™ KV260, Versal™ and Alveo cards. It scales to meet the requirements of many diverse applications in terms of throughput, latency, scalability, and power.
AMD provides pre-built platforms integrating the DPU engine for both edge and data-center cards. These pre-built platforms allow data-scientists to start developping and testing their models without any need for HW development expertise.

# **Vitis AI Model Zoo**
**Model Zoo Details and Performance**
All the models in the Model Zoo are deployed on AMD adaptable hardware with Vitis AI and the Vitis AI Library. The performance benchmark data includes end-to-end throughput and latency for each model, targeting various boards with varied DPU configurations.
To make the job of using the Model Zoo a little easier, we have provided a downloadable spreadsheet and an online table that incorporates key data about the Model Zoo models. The spreadsheet and tables include comprehensive information about all models, including links to the original papers and datasets, source framework, input size, computational cost (GOPs), and float and quantized accuracy. 
The Vitis AI Model Zoo includes optimized deep learning models to speed up the deployment of deep learning inference on adaptable AMD platforms. 

# **Resnet**

ResNet (Residual Network), introduced by researchers at Microsoft Research in 2015, fundamentally addressed one of the most critical challenges in deep learning: the degradation problem in very deep neural networks. As convolutional neural networks (CNNs) became deeper, it was observed that simply stacking more layers did not necessarily improve accuracy; instead, training error often increased due to vanishing gradients and optimization difficulties. ResNet proposed a novel architectural concept called residual learning, where instead of learning a direct mapping H(x), the network learns a residual function 
```
F(x)=H(x)−x.
```
This is implemented through skip connections (identity shortcuts) that bypass one or more layers and add the input directly to the output of stacked convolutional layers. Mathematically, the block output becomes y=F(x)+x, ensuring that gradients can flow directly through identity paths during backpropagation. This significantly stabilizes training and enables extremely deep architectures such as ResNet-50, ResNet-101, and ResNet-152.

The architectural backbone of ResNet consists of stacked residual blocks, each typically containing two or three convolutional layers followed by batch normalization and ReLU activation. In deeper variants (like ResNet-50 and beyond), a bottleneck design (1×1 → 3×3 → 1×1 convolutions) is used to reduce computational complexity while preserving representational power. This design makes ResNet computationally efficient despite its depth.

Advantages of ResNet Over Other Classification Models
When compared with earlier architectures such as VGGNet and AlexNet, ResNet offers several fundamental improvements:
    
1. Elimination of Degradation Problem
Traditional deep CNNs suffer from performance saturation and degradation as depth increases. ResNet’s identity shortcut allows the model to approximate an identity mapping easily, ensuring that deeper networks perform at least as well as their shallower counterparts.

2. Improved Gradient Flow
Skip connections create direct gradient pathways, mitigating the vanishing gradient issue common in very deep networks. This enables stable training of networks exceeding 100 layers.

3. Better Generalization
Residual learning encourages feature refinement rather than complete feature re-learning at each layer. This leads to improved generalization on large-scale datasets such as ImageNet.

4. Computational Efficiency (in Bottleneck Versions)
Compared to VGGNet, which relies heavily on large stacks of 3×3 convolutions with high parameter counts, ResNet-50 achieves superior accuracy with significantly fewer parameters and lower computational overhead.

5. Transfer Learning Superiority
Due to its deep hierarchical feature extraction and stable training behavior, ResNet has become one of the most widely adopted backbone networks in detection and segmentation frameworks such as Faster R-CNN and Mask R-CNN.

6. Scalability
The residual block is modular. Depth can be increased systematically without architectural redesign, unlike earlier CNNs where deeper models required careful manual tuning.
In essence, ResNet shifted the paradigm from simply “increasing depth” to “enabling depth through identity mapping,” making ultra-deep learning practically feasible.
Convolution Layer – Conceptual Understanding

A convolution layer is the fundamental building block of CNN architectures like ResNet. It operates by applying a small learnable kernel (e.g., 3×3 or 5×5) across the spatial dimensions of the input image. This kernel performs an element-wise multiplication followed by summation, generating a feature map that captures local spatial patterns such as edges, textures, or shapes.
If the input image is represented as X and the kernel as K, the convolution operation at spatial location (i,j) can be expressed as:
```
Y(i,j)=m∑​n∑​X(i+m,j+n)⋅K(m,n)
```
Key components include:
    • Stride: Controls how much the filter shifts.
    • Padding: Preserves spatial dimensions.
    • Number of Filters: Determines output depth (feature channels).


In early layers, convolutional filters typically detect low-level features (edges, gradients). In deeper layers (as in ResNet), filters capture higher-level semantic features such as object parts or entire object representations.
ResNet represents a structural breakthrough in deep learning architecture design. While earlier CNNs like AlexNet demonstrated the power of convolutional hierarchies and VGGNet emphasized depth, ResNet provided the mathematical and architectural mechanism necessary to train extremely deep networks effectively. The introduction of residual connections not only improved classification accuracy but also became a foundational concept influencing modern architectures, including DenseNet, EfficientNet, and transformer-based vision models.
# DPU IP Details and System Integration
**About the DPU IP**

AMD uses the acronym D-P-U to identify soft accelerators that target deep-learning inference. These “D eep Learning P rocessing U nits” are a vital component of the Vitis AI solution. This (perhaps overloaded) term can refer to one of several potential accelerator architectures covering multiple network topologies.
A DPU comprises elements available in the AMD programmable logic fabric, such as DSP, BlockRAM, UltraRAM, LUTs, and Flip-Flops, or may be developed as a set of microcoded functions that are deployed on the AMD AI Engine, or “AI Engine” architecture. Furthermore, in the case of some applications, the DPU is likely to be comprised of programmable logic and AI Engine array resources.
An example of the DPUCZ, targeting Zynq™ Ultrascale+™ devices is displayed in the following image:
 
**Features and Architecture of the Zynq Ultrascale+ DPUCZ**
Vitis AI provides the DPU IP and the required tools to deploy both standard and custom neural networks on AMD adaptable targets:
Vitis AI DPUs are general-purpose AI inference accelerators. A single DPU instance in your design can enable you to deploy multiple CNNs simultaneously and process multiple streams simultaneously. The Processing depends on the DPU having sufficient parallelism to support the combination of the networks and the number of streams. Multiple DPU instances can be instantiated per device. The DPU can be scaled in size to accommodate the requirements of the user.

The Vitis AI DPU architecture is called a “Matrix of (Heterogeneous) Processing Engines.” While on the surface, Vitis AI DPU architectures have some visual similarity to a systolic array; the similarity ends there. DPU is a micro-coded processor with its Instruction Set Architecture. Each DPU architecture has its own instruction set, and the Vitis AI Compiler compiles an executable .Xmodel to deploy for each network. The DPU executes the compiled instructions in the .Xmodel. The Vitis AI Runtime addresses the underlying tasks of scheduling the inference of multiple networks, multiple streams, and even multiple DPU instances. The mix of processing engines in the DPU is heterogeneous, with the DPU having different engines specialized for different tasks. For instance, CONV2D operators are accelerated in a purpose-built PE, while another process depthwise convolutions.

You might ask that why not go with the traditional hls4ml or FINN Flow why go on with the Vitis-AI workflow why is this prefered
One advantage of this architecture is that there is no need to load a new bitstream or build a new hardware platform while changing the network. This is an important differentiator from Data Flow accelerator architectures that are purpose-built for a single network. That said, both the Matrix of Processing Engines and Data Flow architectures have a place in AMD designs. 

**Vitis AI DPU IP and Reference Designs**

Today, AMD DPU IPs are not incorporated into the standard Vivado™ IP catalog and instead, the DPU IP is released embedded in a reference design. Users can start with the reference design and modify it to suit their requirements. The reference designs are fully functional and can be used as a template for IP integration and connectivity as well as Linux integration.
The DPU IP is also is released as a separate download that can be incorporated into a new or existing design by the developer.
DPU Nomenclature
There are a variety of different DPUs available for different tasks and AMD platforms. The following decoder helps extract the features, characteristics, and target hardware platforms from a given DPU name.
 
**Historic DPU Nomenclature**
As of the Vitis™ 1.2 release, the historic DPUv1/v2/v3 nomenclature was deprecated. To better understand how these historic DPU names map into the current nomenclature, refer to the following table:
 
**DPU Options**
**Zynq™ UltraScale+™ MPSoC: DPUCZDX8G**
The DPUCZDX8G IP has been optimized for Zynq UltraScale+ MPSoC. You can integrate this IP as a block in the programmable logic (PL) of the selected Zynq UltraScale+ MPSoCs with direct connections to the processing system (PS). The DPU is user-configurable and exposes several parameters which can be specified to optimize PL resources or customize enabled features.


**Zynq UltraScale＋ MPSoC DPU TRD**
The Xilinx Deep Learning Processor Unit(DPU) is a configurable computation engine dedicated for convolutional neural networks. The degree of parallelism utilized in the engine is a design parameter and application. It includes a set of highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others.
Features
    • One AXI slave interface for accessing configuration and status registers.
    • One AXI master interface for accessing instructions.
    • Supports configurable AXI master interface with 64 or 128 bits for accessing data depending on the target device.
    • Supports individual configuration of each channel.
    • Supports optional interrupt request generation.
    • Some highlights of DPU functionality include:
        ◦ Configurable hardware architecture includes: B512, B800, B1024, B1152, B1600, B2304, B3136, and B4096 
        ◦ Maximum of three cores 
        ◦ Convolution and deconvolution 
        ◦ Depthwise convolution 
        ◦ Max poolling 
        ◦ Average poolling 
        ◦ ReLU, RELU6, Leaky ReLU, Hard Sigmoid, and Hard Swish 
        ◦ Concat 
        ◦ Elementwise-sum and Elementwise-multiply 
        ◦ Dilation 
        ◦ Reorg 
        ◦ Correlation 1D and 2D 
        ◦ Argmax and Max along channel dimension 
        ◦ Fully connected layer 
        ◦ Softmax 
        ◦ Bach Normalization 
        ◦ Split 
        
**Hardware Architecture**
The detailed hardware architecture of the DPU is shown in the following figure. After start-up, the DPU fetches instructions from off-chip memory to control the operation of the computing engine. The instructions are generated by the DNNC where substantial optimizations have been performed. On-chip memory is used to buffer input, intermediate, and output data to achieve high throughput and efficiency. The data is reused as much as possible to reduce the memory bandwidth. A deep pipelined design is used for the computing engine. The processing elements (PE) take full advantage of the finegrained building blocks such as multipliers, adders and accumulators in Xilinx devices.

There are three dimensions of parallelism in the DPU convolution architecture - pixel parallelism, input channel parallelism, and output channel parallelism. The input channel parallelism is always equal to the output channel parallelism. The different architectures require different programmable logic resources. The larger architectures can achieve higher performance with more resources. The parallelism for the different architectures is listed in the table.

