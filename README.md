# weather-robustness
Code for the "Evaluating and Increasing Segmentation Robustness in CARLA" paper accepted for WAISE(SAFECOMP 2023) workshop.

Abstract:Model robustness is a crucial property in safety-critical applications such as autonomous driving and medical diagnosis. 
In this paper, we use the CARLA simulation environment to evaluate the robustness of various architectures for semantic segmentation to adverse environmental changes. 
Contrary to previous work, the environmental changes that we test the models against are not applied to existing images, but rendered directly in the simulation, enabling more realistic robustness tests. 
Surprisingly, we find that Transformers provide only slightly increased robustness compared to some CNNs. 
Furthermore, we demonstrate that training on a small set of adverse samples can significantly improve the robustness of most models. 
The code and supplementary results for our experiments are available online.