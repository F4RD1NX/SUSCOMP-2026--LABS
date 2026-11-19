Week 5 Lab Notes – Measuring Software Energy Consumption
Student: Md Shafiul Karim
Course: Sustainable Computing
Operating system: Windows
CPU: Intel Core i5-10300H
GPU: NVIDIA GeForce GTX 1650 Ti
Python environment: suscomp-25
1. Introduction
The aim of this lab was to measure the energy used by different software workloads. I compared pure Python with NumPy, and PyTorch running on the CPU and GPU. I also compared three energy meters to understand how their measurement scopes affect the results.
2. Method
I used the provided workload.py, measure.py and analyse.py scripts. Each workload ran for approximately 30 seconds, after an unmeasured warm-up.
Meter	Measurement scope	Mode
CodeCarbon	CPU	CPU:windows_emi
NVIDIA	GPU board	NVML energy counter
Battery	Whole laptop, including screen	Battery fuel gauge


The meters measure different parts of the computer, so their energy values are not directly interchangeable. The battery measurements were taken with the laptop unplugged.

I completed at least five runs for each workload. Some tests have six runs because I performed an extra repetition. The initial 10-second CodeCarbon idle check is also included in the data.
3. Part A – Pure Python vs. NumPy
I measured idle, pure Python matrix multiplication and NumPy matrix multiplication using CodeCarbon.
Workload	Mean energy (J)	Mean power (W)	Throughput (GFLOP/s)
Idle	364.76	14.24	—
Pure Python	824.41	27.22	0.02
NumPy	1,024.37	33.72	28.52


Result: NumPy used more average power than pure Python, but completed far more work in the same time. Its throughput was approximately 1,426 times higher based on the rounded reported values. This shows why energy efficiency should be evaluated together with the amount of work completed, not only by looking at power.

The idle average includes the initial 10-second check, which was shorter than the other runs.
4. Part B – PyTorch CPU vs. GPU
I ran the same PyTorch matrix multiplication workload on the CPU and on the NVIDIA GPU using CUDA. The CPU and GPU workloads produced matching checksums.
Workload	Meter	Mean energy (J)	Mean power (W)	Throughput (GFLOP/s)
PyTorch CPU	CodeCarbon	918.17	26.26	174.98
PyTorch GPU	NVIDIA	1,482.98	43.81	2,551.84


Result: The GPU achieved approximately 14.6 times the throughput of the CPU. The analysis reported approximately 180 mJ/GFLOP for PyTorch CPU and 19.4 mJ/GFLOP for PyTorch GPU.

However, the CPU and GPU values come from different meters and measurement scopes: CodeCarbon measured CPU energy, while NVIDIA measured GPU-board energy. Therefore, this is not a complete whole-system energy comparison.
5. Part C – Repeating Part A with the Battery Meter
I repeated the idle, Python and NumPy workloads with the laptop unplugged. The battery meter measures the whole laptop, not only the CPU.
Workload	Mean energy (J)	Mean power (W)
Idle	725.01	23.89
Pure Python	823.52	27.04
NumPy	745.25	24.35


Measurement issue: The sixth NumPy battery run reported 200.39 J (6.56 W), while the other five runs were between 778.43 J and 926.04 J. I kept this reading in the dataset instead of removing it, but it is a clear outlier and makes the NumPy average less reliable. The battery fuel-gauge reading may have been unstable during that run.
6. Comparison and Discussion
The results show that optimized libraries and GPU acceleration can greatly increase performance. NumPy was much faster than pure Python, and PyTorch on the GPU achieved the highest throughput.

The meters reported different power values because they measure different hardware scopes:
- CodeCarbon: CPU energy.
- NVIDIA: GPU-board energy.
- Battery: whole-laptop energy, including the display and other components.
For this reason, the readings should not be compared as if they measured exactly the same thing. The battery NumPy outlier also shows that measurement tools can produce unstable readings.

The calculations used in the analysis include:
- Average power: \(P = E/t\)
- Energy in watt-hours: \(Wh = J/3600\)
- Energy above idle: \(E_{workload} - P_{idle} \times t\)
- Energy efficiency: energy consumed divided by useful work completed, such as joules per GFLOP.
7. Reflection and Conclusion
The most interesting result was how much faster NumPy was than pure Python. This demonstrates the benefit of optimized numerical libraries. The GPU also achieved much higher throughput than the CPU for the PyTorch matrix multiplication workload.

I learned that higher power does not automatically mean worse energy efficiency. A program may use more power but finish much more work in the same time. I also learned that the measurement scope and reliability of the meter must be considered before drawing conclusions.

Overall, NumPy and GPU acceleration improved computational performance substantially, while the different meters and the battery outlier showed the importance of careful measurement and interpretation.
8. Files and Reproducibility
The raw measurements are saved in runs.csv. The lab used the provided scripts:
- workload.py
- measure.py
- analyse.py
The analysis was generated with:
python analyse.py runs.csv --baseline python
All 46 recorded measurements were retained, including extra repetitions and the battery NumPy outlier.
