Week 5 Lab Notes --- Measuring Software Energy Consumption
Student: Md shafiul karim
Course: Sustainable Computing
Lab: Week 5 --- Measuring Software Energy Consumption
Operating system: Windows
Processor: Intel Core i5-10300H (Intel64 Family 6 Model 165 Stepping
2)
GPU: NVIDIA GeForce GTX 1650 Ti
Python environment: suscomp-25
1. Introduction
The purpose of this lab was to measure and compare the energy
consumption of different software implementations while they performed
computational workloads. I compared a pure Python matrix multiplication
workload with NumPy, and I also compared PyTorch matrix multiplication
running on the CPU and NVIDIA GPU. I used different meters to understand
how the measurement scope affects the results.

The main questions were whether optimized libraries perform more work
for the same amount of time, whether GPU acceleration improves
performance, and how consistently different energy meters report
consumption.
2. Method
I used the lab-provided workload.py, measure.py, and analyse.py
scripts. The workload ran for approximately 30 seconds per measurement.
I performed warm-up runs before the measured tests and kept the laptop
unplugged for the battery-meter measurements.

The experiment used these workload labels:
- idle: no computational work; used as the baseline.
- python: pure Python matrix multiplication (python-matmul).
- numpy: NumPy matrix multiplication (numpy-matmul).
- torch-cpu: PyTorch matrix multiplication on the CPU.
- torch-gpu: PyTorch matrix multiplication on the CUDA-enabled
  NVIDIA GPU.
Meters and their measurement scope
  Meter             Scope             Mode reported by    What it measures
the tool            
  CodeCarbon        CPU (CodeCarbon)  CPU:windows_emi   CPU energy on
Windows, using
the energy
interface exposed
by the system

  NVIDIA            GPU board         Measured (NVML      Energy reported
energy counter)     by the NVIDIA GPU
energy counter
Battery           Whole laptop      Measured (fuel      Whole-laptop
(battery)         gauge)              battery energy,
including the
display and other
system components
The meters do not measure the same hardware scope. In particular,
CodeCarbon's CPU reading and NVIDIA's GPU-board reading should not be
treated as directly interchangeable. The battery meter measures the
whole laptop, so its readings include components beyond the CPU or GPU.
Repetitions and data handling
The formal workload groups contain five or more measured runs. Some
groups contain six runs because I performed an additional repetition. I
kept all valid recorded runs rather than deleting extra measurements.
The CodeCarbon idle group also contains the initial 10-second meter
check, in addition to the five approximately 30-second idle runs. That
short check is shown separately in the raw table and included in the
analysis output's six-run idle statistics.

The CSV contains 46 valid measurement records in total. Warm-up workload
runs were not recorded as measurements.
3. Part A --- Pure Python compared with NumPy
For Part A, I measured idle, pure Python matrix multiplication, and
NumPy matrix multiplication with CodeCarbon. The idle readings provide a
baseline for the machine while no benchmark workload was running.
Idle --- CodeCarbon
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                   (W) 

        1      10.286       200.78       0.0558       19.52             0

        2      30.127       486.37       0.1351       16.14             0

        3      30.167       415.47       0.1154       13.77             0

        4      30.134       305.69       0.0849       10.14             0

        5      30.127       436.47       0.1212       14.49             0

        6      30.148       343.80       0.0955       11.40             0

 **Mean**   **26.83**   **364.76**   **0.1013**   **14.24**       **0.0**

   **SD**    **8.11**   **103.27**   **0.0287**    **3.37**       **0.0**

Pure Python matrix multiplication --- CodeCarbon
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                   (W) 

        1      30.518       840.78       0.2335       27.55           198

        2      30.221       946.05       0.2628       31.30           183

        3      30.211       765.48       0.2126       25.34           213

        4      30.270       838.97       0.2330       27.72           195

        5      30.241       730.79       0.2030       24.17           201

 **Mean**   **30.29**   **824.41**   **0.2290**   **27.22**     **198.0**

   **SD**    **0.13**    **82.94**   **0.0230**    **2.73**      **10.8**

NumPy matrix multiplication --- CodeCarbon
      Run    Duration    Energy (J)  Energy (Wh)  Avg. power     Repetitions
                  (s)                                    (W) 

        1      30.281       1069.38       0.2971       35.32         285,019

        2      30.460       1075.58       0.2988       35.31         238,052

        3      30.450       1008.99       0.2803       33.14         205,674

        4      30.267       1008.33       0.2801       33.31         276,520

        5      30.436        959.57       0.2665       31.53         232,492

 **Mean**   **30.38**   **1024.37**   **0.2845**   **33.72**   **247,551.4**

   **SD**    **0.10**     **48.33**   **0.0134**    **1.61**    **32,839.5**

Part A interpretation
The mean CodeCarbon power was 14.24 W for the idle group, 27.22 W for
pure Python, and 33.72 W for NumPy. NumPy used more average power than
pure Python during the fixed-duration test, but it completed far more
matrix-multiplication repetitions.

The mean NumPy throughput reported by the analysis was approximately
28.52 GFLOP/s, compared with approximately 0.02 GFLOP/s for pure Python.
This shows that higher instantaneous power does not necessarily mean
worse energy efficiency: the amount of useful work completed must also
be considered.

The initial 10-second idle check is shorter than the other idle
measurements. It is included in the table because it is present in
runs.csv; it should be remembered when interpreting the idle mean and
standard deviation.
4. Part B --- PyTorch CPU compared with PyTorch GPU
I used the same PyTorch matrix-multiplication workload with the device
set to cpu and cuda. I first checked that PyTorch detected the
NVIDIA GPU and completed warm-up runs on both devices. The checksums
reported by the CPU and GPU workloads matched, which indicates that they
completed the same computational task.

The CPU runs were measured with CodeCarbon. The GPU runs were measured
with the NVIDIA meter. Because the two meters cover different hardware
scopes, their energy readings are not a whole-system CPU-versus-GPU
comparison.
PyTorch CPU --- CodeCarbon
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                   (W) 

        1      36.889      1089.12       0.3025       29.52           399

        2      33.185       938.16       0.2606       28.27           328

        3      34.671       979.46       0.2721       28.25           262

        4      33.287       846.83       0.2352       25.44           313

        5      36.964       875.42       0.2432       23.68           214

        6      34.808       780.01       0.2167       22.41           320

 **Mean**   **34.97**   **918.17**   **0.2550**   **26.26**     **306.0**

   **SD**    **1.66**   **108.98**   **0.0303**    **2.86**      **62.9**

PyTorch GPU --- NVIDIA meter
      Run    Duration    Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                    (W) 

        1      33.479       1486.09       0.4128       44.39         4,598

        2      34.018       1475.52       0.4099       43.38         4,347

        3      33.612       1477.91       0.4105       43.97         4,578

        4      33.996       1485.56       0.4127       43.70         4,429

        5      33.907       1482.91       0.4119       43.74         4,503

        6      34.119       1489.87       0.4139       43.67         4,285

 **Mean**   **33.86**   **1482.98**   **0.4119**   **43.81**   **4,456.7**

   **SD**    **0.25**      **5.39**   **0.0015**    **0.34**     **125.8**

Part B interpretation
The analysis reported mean throughput of approximately 174.98 GFLOP/s
for PyTorch CPU and 2,551.84 GFLOP/s for PyTorch GPU. The GPU throughput
was therefore about 14.6 times higher in these measurements.

The reported energy per GFLOP was approximately 180 mJ for PyTorch CPU
and 19.4 mJ for PyTorch GPU. This suggests that the GPU completed the
measured matrix-multiplication work much more efficiently in terms of
the energy reported by its own meter. However, this comparison has an
important limitation: the CPU result comes from CodeCarbon's CPU scope,
while the GPU result comes from the NVIDIA GPU-board counter. A complete
system-level comparison would require measuring the whole system
consistently for both runs.

The six GPU measurements were relatively consistent: their average power
was 43.81 W with a standard deviation of 0.34 W. The CPU measurements
varied more, with average power of 26.26 W and a standard deviation of
2.86 W.
5. Part C --- Repeating Part A with the battery meter
For Part C, I repeated the idle, pure Python, and NumPy workloads using
the battery meter while the laptop was unplugged. This meter measures
the whole laptop rather than only the CPU.
Idle --- Battery meter
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                   (W) 

        1      30.406       604.01       0.1678       19.87             0

        2      30.414       707.94       0.1967       23.28             0

        3      30.261       685.27       0.1904       22.65             0

        4      30.349       869.15       0.2414       28.64             0

        5      30.242       755.55       0.2099       24.98             0

        6      30.399       728.14       0.2023       23.95             0

 **Mean**   **30.35**   **725.01**   **0.2014**   **23.89**       **0.0**

   **SD**    **0.08**    **87.44**   **0.0243**    **2.89**       **0.0**

Pure Python matrix multiplication --- Battery meter
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power   Repetitions
                  (s)                                   (W) 

        1      30.439       726.23       0.2017       23.86           127

        2      30.385       839.79       0.2333       27.64           103

        3      30.354       743.76       0.2066       24.50           133

        4      30.582       963.26       0.2676       31.50           105

        5      30.325       896.63       0.2491       29.57            97

        6      30.649       771.45       0.2143       25.17           123

 **Mean**   **30.46**   **823.52**   **0.2288**   **27.04**     **114.7**

   **SD**    **0.13**    **93.46**   **0.0260**    **3.06**      **14.8**

NumPy matrix multiplication --- Battery meter
      Run    Duration   Energy (J)  Energy (Wh)  Avg. power     Repetitions
                  (s)                                   (W) 

        1      30.569       778.43       0.2162       25.46         156,927

        2      30.491       926.04       0.2572       30.37         160,816

        3      30.511       857.62       0.2382       28.11         155,198

        4      30.628       833.85       0.2316       27.22         157,126

        5      30.853       875.18       0.2431       28.37         167,892

        6      30.562       200.39       0.0557        6.56         248,824

 **Mean**   **30.60**   **745.25**   **0.2070**   **24.35**   **174,463.8**

   **SD**    **0.13**   **271.30**   **0.0754**    **8.86**    **36,710.2**

Part C interpretation
The mean whole-laptop power was 23.90 W for idle, 27.04 W for pure
Python, and 24.35 W for NumPy. These results differ from the CodeCarbon
readings because the battery meter measures the entire laptop, including
the screen and other components, rather than only CPU energy.

One NumPy battery reading is a clear anomaly. The sixth run reported
200.39 J and 6.56 W, while the other five NumPy runs were between 778.43
J and 926.04 J, with average power between 25.46 W and 30.37 W. The
sixth run also reported more repetitions than the other runs. I retained
this result in the dataset instead of silently removing it, but it
increases the NumPy battery group's standard deviation and makes its
mean less representative. The fuel-gauge measurement may have been
unstable or affected by the way the laptop reported battery energy
during that run.
6. Comparison of the meters
The mean power results from the repeated tests are summarized below.

  Test            Battery meter (W)   CodeCarbon (W)   NVIDIA meter (W)
  Idle                        23.90          14.24*                ---
Pure Python                 27.04            27.22                ---
NumPy                       24.35            33.72                ---
PyTorch CPU                   ---            26.26                ---
PyTorch GPU                   ---              ---              43.81

* The CodeCarbon idle mean includes the initial 10-second meter check
and five approximately 30-second idle runs.

The table shows why it is important to state the scope of each meter.
The battery meter reports whole-laptop consumption, while CodeCarbon and
NVIDIA report CPU and GPU-board measurements respectively. Their values
can differ even when measuring similar workloads because they cover
different parts of the computer and use different measurement methods.
7. Energy calculations and interpretation
The basic calculations used in this lab are:
- Average power: P = E / t, where E is energy in joules and t is
  time in seconds.
- Conversion from joules to watt-hours: Wh = J / 3600.
- Energy above idle for a workload:
  E_workload − P_idle × t_workload.
- Energy efficiency: energy consumed divided by the amount of useful
  work completed, for example joules per GFLOP.
Energy per unit of work is useful because the workloads do not all
complete the same number of operations in the same time. A workload can
draw more power but still use less energy per operation if it completes
much more work.

The analysis output reported these summary values:

  Test                         Mean power (W)   Throughput (GFLOP/s)             Energy per GFLOP
  Pure Python (CodeCarbon)              27.22                   0.02   approximately 1.21 × 10³ J
NumPy (CodeCarbon)                    33.72                  28.52         approximately 1.21 J
PyTorch CPU (CodeCarbon)              26.26                 174.98        approximately 0.180 J
PyTorch GPU (NVIDIA)                  43.81               2,551.84       approximately 0.0194 J
Pure Python (Battery)                 27.04                   0.01   approximately 2.13 × 10³ J
NumPy (Battery)                       24.35                  20.10         approximately 1.33 J

The battery NumPy energy-per-GFLOP summary is affected by the anomalous
sixth run. It should therefore be interpreted with caution.
8. Reflection
The most noticeable result was the difference between pure Python and
optimized numerical libraries. NumPy completed vastly more
matrix-multiplication work in the same approximate 30-second period.
This shows the benefit of using optimized libraries for computational
tasks instead of implementing every operation in ordinary Python loops.

The PyTorch GPU test had the highest throughput. It achieved
approximately 2,551.84 GFLOP/s, compared with 174.98 GFLOP/s on the CPU.
This demonstrates the advantage of GPU acceleration for a workload that
can use parallel computation effectively. The GPU's measured board power
was higher, but its reported energy per GFLOP was lower. Since the CPU
and GPU meters have different scopes, this result is an indication
rather than a perfectly controlled whole-system energy comparison.

The different meters did not produce directly comparable power readings.
CodeCarbon measured CPU energy, NVIDIA measured GPU-board energy, and
the battery meter measured the whole laptop. The battery results were
also less stable for the NumPy workload because one run was much lower
than the other five. This showed me that measurement tools and their
limitations matter when drawing conclusions about software energy
consumption.

Overall, the experiment showed that performance and energy efficiency
should be considered together. The fastest implementation is not
automatically the one with the lowest total energy in every situation,
and the measurement scope must be understood before comparing results.
9. Conclusion
The experiment demonstrated that optimized libraries and GPU
acceleration can greatly increase computational throughput. NumPy was
much faster than pure Python, and PyTorch on the NVIDIA GPU achieved the
highest measured throughput. The measurements also showed that energy
readings depend on the meter's scope and reliability. For this reason,
energy-per-work calculations and clear descriptions of the measurement
method are important when evaluating software sustainability.
10. Files and reproducibility
The raw measurements are stored in runs.csv. The experiment was
performed using the lab-provided scripts:
- workload.py
- measure.py
- analyse.py
The analysis was generated with:
python analyse.py runs.csv --baseline python
The warm-up runs were not included as measured records. All 46 valid
records in the supplied CSV were retained in this report, including the
extra repetitions and the battery NumPy outlier.
