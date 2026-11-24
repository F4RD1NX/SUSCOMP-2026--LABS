# Week 9 – Measuring Cloud Energy Consumption and Emissions

**Student:** Md Shafiul Karim  
**Environment:** Google Colab with GPU runtime  
**Date:** 24 November 2026

## 1. Introduction

The purpose of this lab was to investigate energy consumption and carbon emissions when running computations in a cloud environment. I measured GPU and CPU performance, checked the location and carbon intensity of the Colab server, tested network transfer speeds, and compared machine learning training on CPU and GPU.

## 2. Cloud Environment and Hardware

The Colab environment provided:

| Component | Result |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU memory | 15,360 MiB |
| GPU power limit | 70 W |
| GPU idle power | 9 W |
| CPU | Intel Xeon CPU @ 2.00 GHz |
| Virtual CPUs | 2 |
| Host CPU threads | 56 |
| Estimated CPU power at full load | 7.3 W |

The GPU initially operated in P8 power-saving mode, consuming approximately 9 W while idle.

After creating a CUDA context, its state changed to P0 and its power consumption increased to 26.94 W. This shows that an allocated GPU can consume additional energy even when it is not performing useful calculations.

## 3. GPU Startup Cost

The measured startup times were:

| Operation | Time |
|---|---:|
| Check CUDA availability | 0.00015 s |
| First GPU use | 0.00064 s |
| Second GPU use | 0.00010 s |

The first GPU operation took longer than the second because the GPU needs to initialize its CUDA context. In this session, the measured times were very small.

## 4. CPU and GPU Performance Comparison

I compared matrix multiplication on the CPU and GPU.

| Measurement | CPU | GPU |
|---|---:|---:|
| Execution time | 20.953 s | 6.144 s |
| Operations | 2.5 TFLOP | 25 TFLOP |
| Performance | 119 GFLOP/s | 4,069 GFLOP/s |
| Energy | 153 J | 424 J |
| Energy per GFLOP | 61.36 mJ | 16.97 mJ |
| Average power | 7.3 W estimated | 69.1 W measured |

The GPU achieved approximately 34 times higher throughput than the CPU. It also used less energy per GFLOP, making it more efficient for this type of computation.

However, the GPU completed ten times more operations than the CPU, so comparing total energy alone would be misleading.

The GPU's average power was 69.1 W, very close to its 70 W power limit.

### Carbon emissions from matrix multiplication

The Colab region had a carbon intensity of 366 gCO₂e/kWh.

Using:

`Emissions = Energy (kWh) × Carbon intensity (gCO₂e/kWh)`

- CPU: 153 J = 0.0000425 kWh, resulting in approximately **0.0156 gCO₂e**.
- GPU: 424 J = 0.0001178 kWh, resulting in approximately **0.0431 gCO₂e**.

The GPU produced more total emissions in this test because it performed significantly more work.

### Comparison with Week 5

In Week 5, my laptop's PyTorch GPU achieved approximately 2,552 GFLOP/s and used 19.4 mJ/GFLOP.

The Colab Tesla T4 achieved 4,069 GFLOP/s and 16.97 mJ/GFLOP. Therefore, the Colab GPU performed better in this particular test and used less energy per GFLOP.

The Week 5 laptop GPU's average power was approximately 43.81 W, compared with 69.1 W for the Colab GPU.

The measurements use different hardware and measurement environments, so the comparison is approximate.

## 5. Colab Server Location and Carbon Intensity

The Colab server's IP information showed:

- City: Singapore
- Country: Singapore
- Cloud provider: Google LLC
- Region: `asia-southeast1`

Google's 2025 regional carbon intensity data reported:

| Measurement | Result |
|---|---:|
| Grid carbon intensity | 366 gCO₂e/kWh |
| Google carbon-free energy (CFE) | 5% |

### Regional comparison

From the regional table:

- Lowest carbon intensity: Montréal, Canada – 11.36 gCO₂e/kWh.
- Highest carbon intensity: Johannesburg, South Africa – 641.42 gCO₂e/kWh.
- Difference: 630.06 gCO₂e/kWh.

The closest listed Google Cloud region to Slovenia is Milan, Italy (`europe-west8`), with approximately 202.03 gCO₂e/kWh.

If the same computation ran in the lowest-carbon region, its emissions would be approximately 96.9% lower than in Singapore, assuming the same energy consumption.

### Live electricity mix

The notebook contains yearly regional averages, but I did not save a live Electricity Maps observation. Therefore, I cannot reliably identify the dominant electricity sources at the exact measurement time or compare a live value with the yearly average.

The notebook's Slovenian reference intensity was set to 250 gCO₂e/kWh. This was the value used for the `g_co2e_ref` calculations.

## 6. Data Transfer Speed

### 6.1 Downloading MNIST to the local computer

The MNIST dataset was downloaded to my Windows computer.

| Measurement | Result |
|---|---:|
| File size | 10.96 MB |
| Download time | 41.38 seconds |
| Download speed | 0.26 MB/s |

### 6.2 Uploading MNIST from the computer to Colab

The notebook recorded:

| Measurement | Result |
|---|---:|
| File size | 11.5 MB |
| Upload time | 488.23 seconds |
| Upload speed | 0.02 MB/s |
| Upload speed in Mbit/s | 0.2 Mbit/s |

This measurement also includes the time spent selecting the file in the upload dialog, so it is an upper bound rather than a precise network transfer time.

### 6.3 Downloading MNIST from Colab to the computer

The notebook used `files.download()` to download the same file back to the local computer.

The download time could not be measured by the notebook, but the download was initiated successfully.

### 6.4 Downloading MNIST directly to Colab

The same dataset was downloaded directly from Google Cloud Storage to the Colab runtime.

| Measurement | Result |
|---|---:|
| File size | 10.96 MB |
| Download time | 1.8 seconds |
| Average speed | 5.96 MB/s |

The Colab download was significantly faster than the local download. This suggests that downloading data directly to the cloud environment can be more efficient than transferring it through a local computer first.

### Comparison

The local download speed was approximately 0.26 MB/s, while Colab achieved 5.96 MB/s.

The Colab download was therefore around 23 times faster.

For machine learning on Colab, it is generally better to download datasets directly from cloud storage located close to the computing region. Data location matters because network distance and routing can affect transfer speed and latency.

## 7. Network Latency and Traceroute

### 7.1 Colab ping

The Colab runtime pinged Google Cloud Storage.

| Measurement | Result |
|---|---:|
| Packets sent | 5 |
| Packets received | 5 |
| Packet loss | 0% |
| Minimum latency | 0.274 ms |
| Average latency | 0.337 ms |
| Maximum latency | 0.482 ms |

The very low latency suggests that the Colab runtime was reaching a nearby Google network endpoint.

### 7.2 Local Windows ping

My local Windows computer produced:

| Measurement | Result |
|---|---:|
| Packets sent | 4 |
| Packets received | 4 |
| Packet loss | 0% |
| Minimum latency | 9 ms |
| Average latency | 9.5 ms |
| Maximum latency | 10 ms |

The local latency was considerably higher than Colab's. This is expected because the local computer communicates through its internet connection, while Colab uses Google's cloud infrastructure.

### 7.3 Colab traceroute

The Colab traceroute reached Google Storage in 14 hops.

The final destination responded with approximately 0.24–0.38 ms latency. Several intermediate hops did not respond, but the destination was still reached successfully.

### 7.4 Local Windows traceroute

The Windows traceroute reached Google Storage in 16 hops.

The route passed through ARNES Slovenia and the GEANT network before reaching Google's Milan network.

The final destination responded with approximately 9–16 ms latency. Some intermediate hops showed higher latency, including a 71 ms response, and one hop timed out.

### Comparison and conclusion

Colab had lower latency and fewer network hops than my local computer.

The results show that the cloud runtime had a much shorter network response time to the Google Storage endpoint. However, traceroute and ping do not reveal the exact physical location of the storage data because Google uses distributed and anycast network infrastructure.

The local and Colab tests also used different IP versions, IPv6 and IPv4 respectively, so they are not perfectly identical comparisons.

## 8. Dataset Download and Loading Time

A larger dataset, YearPredictionMSD, was downloaded from the UCI Machine Learning Repository.

| Measurement | Result |
|---|---:|
| Dataset size | 201.24 MB |
| Download time | Approximately 11 seconds |
| Average download speed | 18.4 MB/s |
| CSV loading time | 10.06 seconds |
| Number of columns | 13 |

The download and CSV loading times were relatively similar.

This shows that transferring a dataset is not always the main bottleneck. Loading and parsing a large dataset can also take considerable time.

For data pipelines, it is important to optimise both network transfer and data loading rather than focusing only on download speed.

## 9. Machine Learning Training Energy Consumption

The final experiment compared CPU and GPU training using two data-loading methods:

- **DataLoader:** Uses PyTorch's DataLoader to fetch samples and build batches.
- **Slices:** Uses shuffled tensor indices and takes batches directly from the tensors.

Each configuration was repeated three times.

The idle measurements were:

| Component | Idle power |
|---|---:|
| CPU | 0.9 W estimated |
| GPU | 31.3 W measured |

### 9.1 Training results

The following table shows the average results across three runs.

| Configuration | Time (s) | GPU utilisation | Total energy (Wh) | Energy per epoch (mWh) |
|---|---:|---:|---:|---:|
| CPU DataLoader | 32.02 | 0.02% | 0.3194 | 31.94 |
| GPU DataLoader | 36.00 | 17.05% | 0.3669 | 36.69 |
| CPU Slices | 30.00 | 0.05% | 0.2994 | 2.99 |
| GPU Slices | 18.11 | 13.69% | 0.1857 | 1.86 |

### 9.2 DataLoader: Was the GPU faster than the CPU?

No. The GPU DataLoader version took approximately 36 seconds, while the CPU version took approximately 32 seconds.

GPU utilisation was only around 17%.

The main reason is that the DataLoader version handles samples through Python and builds batches one at a time. This creates overhead and prevents the GPU from being used efficiently.

The GPU spends time waiting for data rather than performing calculations. For this small model, the CPU can handle the work faster.

### 9.3 Slices: How did performance change?

The slices method improved performance on both devices.

**CPU:**
- Time decreased from 32.02 seconds to 30.00 seconds.
- Energy per epoch decreased from 31.94 mWh to 2.99 mWh.

**GPU:**
- Time decreased from 36.00 seconds to 18.11 seconds.
- Energy per epoch decreased from 36.69 mWh to 1.86 mWh.

The GPU became the better option with the slices method.

It was approximately 40% faster than the CPU and consumed around 38% less energy per epoch.

The slices method is more efficient because it processes batches directly from tensors instead of relying on Python to handle samples individually.

The slices version also ran for 100 epochs instead of 10, so energy per epoch is the more appropriate comparison. Its lower MSE is also partly explained by the larger number of training epochs.

### 9.4 GPU energy during CPU training

Even during CPU-only training, the GPU consumed a significant amount of energy.

| CPU configuration | GPU energy (Wh) | Total energy (Wh) |
|---|---:|---:|
| DataLoader | 0.2786 | 0.3194 |
| Slices | 0.2611 | 0.2994 |

The GPU consumed approximately 87% of the total energy during CPU DataLoader training and 87% during CPU slices training.

This is because the GPU was allocated to the Colab session even though the CPU was performing the training.

The idle GPU power was 31.3 W, which is relatively high for a device that is not doing useful work.

### 9.5 Total energy vs energy above idle

The experiment also calculated energy above idle by subtracting the idle power consumption from the total energy.

| Configuration | Total energy (Wh) | Above-idle energy (Wh) |
|---|---:|---:|
| CPU DataLoader | 0.3194 | 0.0333 |
| GPU DataLoader | 0.3669 | 0.0453 |
| CPU Slices | 0.2994 | 0.0314 |
| GPU Slices | 0.1857 | 0.0239 |

I would report both values, but use total energy as the main result when discussing the complete energy cost of using the Colab session.

Above-idle energy is useful when analysing the additional energy caused by the workload itself.

The difference is important because an allocated GPU can consume energy even when it is not being used.

### 9.6 Carbon emissions from model training

The Colab region's carbon intensity was 366 gCO₂e/kWh.

| Configuration | Colab emissions (gCO₂e) | Slovenian reference emissions (gCO₂e) |
|---|---:|---:|
| CPU DataLoader | 0.1168 | 0.0798 |
| GPU DataLoader | 0.1342 | 0.0917 |
| CPU Slices | 0.1095 | 0.0749 |
| GPU Slices | 0.0679 | 0.0464 |

The GPU slices configuration produced the lowest emissions among the four configurations.

### 9.7 What if the workload ran in the lowest-carbon region?

The lowest carbon intensity in the regional table was Montréal, with 11.36 gCO₂e/kWh.

If the same amount of energy were consumed there, emissions would be approximately 96.9% lower than in the Singapore region.

This demonstrates that the location of cloud computing infrastructure can have a major effect on carbon emissions, even when the amount of computing work remains unchanged.

## 10. Optional Large-Batch Experiment

The lab suggests running:

- `train("cuda", "slices", 100, batch_size=65536)`
- `train("cpu", "slices", 100, batch_size=65536)`

This optional experiment was not performed, so there are no measured results for GPU utilisation, execution time, energy per epoch or MSE with the larger batch size.

I therefore cannot make a reliable numerical comparison for this part.

## 11. Overall Reflection

The experiment demonstrated that cloud computing can provide much higher computational performance, but energy consumption depends on more than just processing speed.

The Tesla T4 GPU performed significantly better than the CPU in matrix multiplication. However, the first machine learning implementation using DataLoader was actually slower on the GPU because Python data-loading overhead limited GPU utilisation.

The slices implementation improved performance and energy efficiency, making the GPU the better choice for this workload.

The network experiments also showed that downloading data directly to Colab was much faster than downloading it to my local computer. Colab also had much lower network latency.

Finally, the carbon intensity of the cloud region had a significant impact on estimated emissions. An allocated GPU also consumed considerable energy while idle, showing why resource allocation and utilisation matter in sustainable computing.

## 12. Conclusion

This lab showed that sustainable cloud computing requires considering performance, energy consumption, network transfer and the electricity mix of the cloud region.

The most efficient training configuration in my experiment was GPU with slices, which achieved the shortest training time and lowest energy per epoch.

Overall, optimising data handling, avoiding unnecessary resource allocation and choosing lower-carbon cloud regions can help reduce the environmental impact of cloud computing.
