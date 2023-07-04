# CVOS-sensor
Title : Real-time multiaxial strain mapping using computer vision integrated optical sensors

Abstract : 
Soft strain sensors pose great potential for emerging human-machine interfaces. However, their real-world applications have been limited due to challenges such as low reproducibility, susceptibility to environmental noise, and short lifetimes, which are attributed to nanotechnologies, including microfabrication techniques. In this study, we present a computer vision-based optical strain (CVOS) sensor system that integrates computer vision with streamlined microfabrication techniques to overcome these challenges and facilitate real-time multiaxial strain mapping. The proposed CVOS sensor consists of an easily fabricated soft silicone substrate with micro-markers and a tiny camera for highly sensitive marker detection. Real-time multiaxial strain mapping allows for measuring and distinguishing complex multi-directional strain patterns, providing the proposed CVOS sensor with higher scalability. Our results indicate that the proposed CVOS sensor is a promising approach for the development of highly sensitive and versatile human-machine interfaces that can operate long-term under real-world conditions.

DOI : 

![fig1](https://github.com/HongSungUk/CVOS-sensor/assets/26831528/e06c19a4-ac4b-4fba-af2b-2cb0104ef3a5)
**Fig. 1. Design and mechanism of CVOS sensor.** a Design (Scale bar: 100 µm). b Strain detection mechanism. c Movement of the micro-marker according to multiaxial strain direction. d Images of micro-markers captured by the optical system to detect the applied tensile strain. e Comparison of micro-marker positions obtained via numerical simulations and measurements.

One of the most significant findings of this study is the multiaxial strain mapping function of the CVOS sensor to detect complex strains that are typically challenging to classify with previously reported strain sensor systems.
![fig6](https://github.com/HongSungUk/CVOS-sensor/assets/26831528/dc457b36-6127-4404-a500-026fe3be50d3)
**Fig. 6. Body motion monitoring system with the CVOS sensor.** a Elbow bending. b Knee bending according to the bending angle. c Wrist bending. d Responses of the CVOS and IMU sensors to forearm rotation. e Detection of complex body motions based on movement direction and angle. f Shoulder rotation motion detection via multiaxial strain mapping.

The uploaded files consist of the development code for the CVOS sensor(OTSS_v5.2_Eng.py) and a test image dataset(test_data.zip). The code encompasses the procedure of detecting micro-markers from the captured images of the sensing part (or, alternatively, the test image dataset) using the optical system and subsequently processing them
