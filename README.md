<h1 align="center">
  Clover-2: Accurate Inference for Regressive Lightweight Speculative Decoding
</h1>

<p align="center">
| <a href="https://github.com/XiaoBin1992/clover"><b>Paper</b></a> |
</p>

---

Clover-2 is an RNN-based draft model designed to achieve comparable accuracy to that of attention decoder layer models while maintaining minimal computational overhead. Clover-2 enhances the model architecture and incorporates knowledge distillation to increase Clover's accuracy and improve overall efficiency. For more details see our [paper](https://arxiv.org/abs/2402.05109).

Despite its RNN architecture, Clover-2 also delivers a maximum {9.3\%} faster speed increase on speculative heads compared to EAGLE.

<div align="center">
  <picture>
  <img src="./figs/result.png" width="80%">
  </picture>
  <br>
  <div align="left" width="80%">
  <em> End-to-end throughput on Vicuan 7B v1.5 (V 7B) and LLaMA3-Instruction 8B (L 8B) with different decoding methods on six tasks. </em>
  </div>
  <br>
</div>

### Setup & InstallationFrom the source

```bash
git clone https://github.com/XiaoBin1992/clover.git
cd clover
pip install -e .
```

### Generate Train Data

You can run the following command to generate the training data.

```bash
python -m clover.ge_data.allocation --outdir [path of data]
```

### Train and EvaluationInference

*clover/stripts* provides examples of .sh files.

### Reference

```

```

### Acknowledgements

This code uses libraries from [EAGLE](https://github.com/SafeAILab/EAGLE), [Medusa](https://github.com/FasterDecoding/Medusa), and [FastChat](https://github.com/lm-sys/FastChat), repository.
