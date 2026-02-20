RESULTS:
For YOLO based Image Identification :
```
sample : ./test_jpeg_yolov3 yolov3_coco_416_tf2 sample_yolov3.jpg
    output : 
I0530 02:07:58.395725 519058 process_result.hpp:44] RESULT: 0   92.2634 134.684 111.956 158.389 0.376607
I0530 02:07:58.395861 519058 process_result.hpp:44] RESULT: 2   111.763 137.506 188.071 183.163 0.995223
I0530 02:07:58.396015 519058 process_result.hpp:44] RESULT: 2   -12.2392        131.704 137.291 253.55  0.994435
I0530 02:07:58.396116 519058 process_result.hpp:44] RESULT: 2   357.697 143.444 417.126 165.011 0.988971
I0530 02:07:58.396205 519058 process_result.hpp:44] RESULT: 2   398.145 132.615 512     254.461 0.988122
I0530 02:07:58.396301 519058 process_result.hpp:44] RESULT: 2   327.152 143.771 358.783 157.925 0.959619
I0530 02:07:58.396389 519058 process_result.hpp:44] RESULT: 2   290.846 138.551 331.462 152.705 0.913988
I0530 02:07:58.396476 519058 process_result.hpp:44] RESULT: 2   205.926 138.871 218.234 146.871 0.773242
I0530 02:07:58.396576 519058 process_result.hpp:44] RESULT: 2   256.975 139.756 269.283 145.986 0.676414
I0530 02:07:58.396661 519058 process_result.hpp:44] RESULT: 2   184.36  137.603 204.052 156.064 0.535515
I0530 02:07:58.396745 519058 process_result.hpp:44] RESULT: 2   271.949 141.241 287.753 147.471 0.498764
I0530 02:07:58.396829 519058 process_result.hpp:44] RESULT: 2   193.414 137.306 208.75  155.768 0.469957
I0530 02:07:58.396914 519058 process_result.hpp:44] RESULT: 2   173.308 137.306 193     155.768 0.319147
```


For YOLO based Real-Time Video identification using Webcam 
refer this google link
```
https://drive.google.com/file/d/1cSM9rjGFbIcV-OHkdyxnrXiWy6aVgUDC/view?usp=sharing
```


For Resnet based Image Classification 

```
sample : ./test_jpeg_classification resnet50_pt sample_classification.jpg
    output :
I0604 12:22:21.709828 1750 demo.hpp:1183] batch: 0 image: sample_classification.jpg
I0604 12:22:21.709856 1750 process_result.hpp:24] r.index 109 brain coral, r.score 0.99577
I0604 12:22:21.710084 1750 process_result.hpp:24] r.index 973 coral reef, r.score 0.00192229
I0604 12:22:21.710193 1750 process_result.hpp:24] r.index 5 electric ray, crampfish, numbfish, torpedo, r.score 0.00116593
I0604 12:22:21.710388 1750 process_result.hpp:24] r.index 392 rock beauty, Holocanthus tricolor, r.score 0.000202608
I0604 12:22:21.710573 1750 process_result.hpp:24] r.index 329 sea cucumber, holothurian, r.score 0.000202608
```

For Resnet based Classification in Real-Time
refer this drive link
```
https://drive.google.com/file/d/1WYrvCRc0r0VpIBs-5dC2FuCz70g1NvWV/view?usp=sharing
```

How does FPGA compare to CPU and GPU acceleration?
FPGA accelerated networks can run upto 90x faster as compared to CPU. FPGA accelerated networks are on par with GPU accelerated networks for throughput critical applications, yet provide support for more custom applications. FPGA accelerated networks are far superior to GPU accelerated networks for latency critical applications such as autonomous driving. 

Performance Benchmarking Results
(NOTE : This is a comparitive study done by AMD Xilinx on alveo cards and this behaviour is generally applied to every other Vitis-AI supported SoC’s)

With the growing number of real-time AI services, latency becomes an important aspect of overall
AI service performance. Unlike GPUs, where there is a significant trade-off between latency and
throughput, xDNNv3 DNN engines can provide both low-latency and high-throughput. In addition, xDNNv3 kernels provide simple Batch=1 interfacing, which reduces complexities in interfacing software by not requiring any queuing software to auto-batch input data to achieve maximum throughput.

Both the below figures show CNN, latency, and throughput benchmarks on Alveo accelerator cards and popular GPU and FPGA platforms. The first figure shows GoogLeNet V1 Batch=1 throughput measured in images per second along the left Y-axis. The number shown above the throughput is measured/reported latency in milliseconds.
Source: https://docs.amd.com/v/u/en-US/wp504-accel-dnns

While GoogLeNet v1 performance is shown for benchmarking purposes, xDNN supports a broad range of CNN networks. Refer to ML Suite documentation (https://github.com/Xilinx/ml-suite) for more information on running other CNN networks.

As shown in the shared performance results, the xDNN processing engine is a high-performance, energy-efficient DNN accelerator that outperforms many common CPU/GPU platforms today for real-time inference workloads. The xDNN processing engine is available through the ML Suite on many cloud environments, such as Amazon AWS / EC2 or Nimbix NX5. It scales seamlessly toon-premises deployment through Xilinx's new Alveo accelerator cards. Xilinx's reconfigurable FPGA silicon allows users to continue receiving new improvements and features through xDNN updates. This allows the user to keep up with changing requirements and evolving networks. For more information about getting started, go to: 
```
https://github.com/Xilinx/ml-suite 
https://www.xilinx.com/applications/megatrends/machine-learning.html
```
Performance and scalability 

(Based on the research article of Kaiping et al)

Performance Analysis 

After deploying, the evaluation of the system performance of the ZCU102 and the acceleration effect of the SCI model was verified by testing. This was also compared to other platform performance. The FPGA accelerated system processed an average of 67.72 frames/s with an average detection latency of 14,767 μs at frame level.
At the IMX274 camera module, RAW10 image data was acquired through the MIPI FMC interface of the board during the test. FPGA process the data and output it to monitor via HDMI 2.0. The real-time detection results are presented in Figure 6, including lowlight enhancement effects. This comparison shows that the FPGA fully improved images under low-light (nighttime) conditions in the real-time detection system.

To measure its performance, the algorithm was implemented and tested on CPU, GPU, and proposed ZCU102 platform. This comparison regarded multiple parameters like the type of data processing, the image size, the frames per second (FPS), power consumption, energy efficiency, PSNR, and SSIM. Detection performance metrics for each platform are summarized in the document. This cross-platform code/table compares their Low-light image enhancement performance. Compared with CPU platform (Intel i7-10700), GPU platform (RTX 3060 (12GB VRAM)) has superior computational power and resource utilization, and is more suited for complex graphics and video data. With INT8 data, the ZCU102 platform is less energy-consuming and capable of edge computing. 

