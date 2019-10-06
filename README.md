# MISSION
[MISSION: Ultra Large-Scale Feature Selection using Count-Sketches](https://arxiv.org/abs/1806.04310)

An ICML 2018 paper by Amirali Aghazadeh\*, Ryan Spring\*, Daniel LeJeune, Gautam Dasarathy, Anshumali Shrivastava, Richard G. Baraniuk

\* These authors contributed equally and are listed alphabetically.

# How-To-Run + Code Versions
* All data files are formatted using the [VW input format](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
0. Build executables by running Makefile
1. Mission Logistic Regression
```
// Hyperparameters
// Size of Top-K Heap
const size_t TOPK = (1 << 14) - 1;

// Size of Count-Sketch Array
const size_t D = (1 << 18) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 5e-1;

./mission_logistic train_data test_data
```

2. Fine-Grained Mission Softmax Regression
```
// Hyperparameters

// Size of Top-K Heap
const size_t TOPK = (1 << 20) - 1;

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 1e-2;

// Length of String Feature Representation
const size_t LEN = 12;

./fine_mission_softmax train_data test_data
```

3. Coarse-Grained Mission Softmax Regression
```
// Hyperparameters

// Size of Top-K Heap
const size_t TOPK = (1 << 22) - 1;

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 1e-1;

// Length of String Feature Representation
const size_t LEN = 12;

./coarse_mission_softmax [train_data_part_1 train_data_part_2 ... train_data_part_n] test_data
```

4. Feature Hashing Softmax Regression
```
// Hyperparameters

// Number of Classes
const size_t K = 193;

// Size of Count-Sketch Array
const size_t D = (1 << 24) - 1;

// Learning Rate
const float LR = 1e-2;

// Length of String Feature Representation
const size_t LEN = 12;

./softmax [train_data_part_1 train_data_part_2 ... train_data_part_n] test_data
```

# Optimizations

* Mission streams in the dataset via Memory-Mapped I/O instead of loading everything directly into memory -\
Necessary for Tera-Scale Datasets
* AVX SIMD optimization for fast Softmax Regression
* The code is currently optimized for the Splice-Site and DNA Metagenomics datasets.

### Mission Softmax Regression
0. Fine-Grained Feature Set - Each class maintains a separate feature set, so there is a top-k heap for each class.
1. Coarse-Grained Feature Set - All the classes share a common set of features, so there is only one top-k heap. -\
Each feature is measured by its L1 Norm for all classes.
2. Data Parallelism - Each worker maintains a separate heap, while aggregating gradients in the same count-sketch.

# Datasets
1. [KDD 2012](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#kdd2012)
2. [RCV1](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary)
3. [Webspam - Trigram](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#webspam)
5. [DNA Metagenomics](http://projects.cbio.mines-paristech.fr/largescalemetagenomics/)
6. [Criteo 1TB](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#criteo_tb)
7. [Splice-Site 3.2TB](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#splice-site)
