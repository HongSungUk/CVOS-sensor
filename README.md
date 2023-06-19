# CVOS-sensor
Title : Real-time multiaxial strain mapping using computer vision integrated optical sensors

Abstract : 
Soft strain sensors play a major role in emerging human–machine interfaces. Most advanced soft strain sensors rely on nanotechnologies, including microfabrication techniques. However, their wide-scale application is limited due to challenges such as low reproducibility, susceptibility to environmental noise, and short lifetimes, which are attributed to their specialized fabrication techniques. In this study, we present a vision-based strain sensing approach that integrates computer vision with streamlined microfabrication techniques to overcome these challenges and facilitate real-time multiaxial strain mapping, which enables the sensing of strain in multiple directions. We developed a computer vision-based optical strain (CVOS) sensor system that uses an easily fabricated soft silicone substrate with micro-markers and a tiny camera as a highly sensitive marker detector. The ability to perform multiaxial strain mapping allows for obtaining various types of deformation information that are difficult to obtain with traditional strain sensors, which in turn provides the proposed sensor with higher scalability than existing sensors. Our results indicate that the proposed CVOS sensor is a promising approach for the development of highly sensitive and versatile human-machine interfaces that can operate long-term under real-world conditions, surpassing traditional strain sensors in terms of scalability with real-time multiaxial strain mapping.

DOI : 

![fig1](https://github.com/HongSungUk/CVOS-sensor/assets/26831528/e06c19a4-ac4b-4fba-af2b-2cb0104ef3a5)
**Fig. 1. Design and mechanism of CVOS sensor.** a Design (Scale bar: 100 µm). b Strain detection mechanism. c Movement of the micro-marker according to multiaxial strain direction. d Images of micro-markers captured by the optical system to detect the applied tensile strain. e Comparison of micro-marker positions obtained via numerical simulations and measurements.

One of the most significant findings of this study is the multiaxial strain mapping function of the CVOS sensor to detect complex strains that are typically challenging to classify with previously reported strain sensor systems.
![fig6](https://github.com/HongSungUk/CVOS-sensor/assets/26831528/dc457b36-6127-4404-a500-026fe3be50d3)
**Fig. 6. Body motion monitoring system with the CVOS sensor.** a Elbow bending. b Knee bending according to the bending angle. c Wrist bending. d Responses of the CVOS and IMU sensors to forearm rotation. e Detection of complex body motions based on movement direction and angle. f Shoulder rotation motion detection via multiaxial strain mapping.

The uploaded files consist of the development code for the CVOS sensor(OTSS_v5.2_Eng.py) and a test image dataset(test_data.zip). The code encompasses the procedure of detecting micro-markers from the captured images of the sensing part (or, alternatively, the test image dataset) using the optical system and subsequently processing them
