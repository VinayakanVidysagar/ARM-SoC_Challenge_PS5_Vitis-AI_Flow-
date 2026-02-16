YOLO V3 MODEL - ARM SoC Challenge Vitis-AI Flow

Source AMD Vitis-AI Github repository > https://github.com/Xilinx/Vitis-AI/tree/3.0


1, compile

    execute the following command:

        sh build.sh

2, copy the compiled executable file and test image to the development board.

   run the executable file.

    sample : ./test_jpeg_solo solo_pt sample_solo.jpg
    output :
I1213 13:36:38.396692 10265 demo.hpp:1183] batch: 0     image: sample_solo.jpg

I1213 13:36:38.446588 10265 process_result.hpp:108] object: 0 refrigerator    0.5786

I1213 13:36:38.490872 10265 process_result.hpp:108] object: 1 dining_table    0.41515

I1213 13:36:38.491019 10265 process_result.hpp:108] object: 2 oven    0.56029

I1213 13:36:38.491158 10265 process_result.hpp:108] object: 3 bowl    0.34969

I1213 13:36:38.491256 10265 process_result.hpp:108] object: 4 refrigerator    0.46503

I1213 13:36:38.491391 10265 process_result.hpp:108] object: 5 orange    0.30419

I1213 13:36:38.491513 10265 process_result.hpp:108] object: 6 chair    0.35602

I1213 13:36:38.491626 10265 process_result.hpp:108] object: 7 orange    0.33824

I1213 13:36:38.491748 10265 process_result.hpp:108] object: 8 sink    0.32094


Performance: 
    ./test_performance_solo solo_pt test_performance_solo.list -s 60 -t <thread> 


 
