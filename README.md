# Segmentation-Expert-Mixture-Regularization : An Adaptive Learning Method for Imbalanced Regression Problems - CAIAC2026

---

## Important implementation notes

To avoid data leakage:

- segmentation thresholds are computed only from the training data,
- model selection is performed only using the training data,
- test targets are never used for segmentation or gating,
- the preliminary global model is used to estimate the target region during inference.

---

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{rashiwa2026ser,
  title={Segmentation Expert-Mixture Regularization: An Adaptive Learning Method for Imbalanced Regression Problems},
  author={Rashiwa, Sarah Elyane and Branco, Paula},
  booktitle={The 39th Canadian Conference on Artificial Intelligence},
  year={2026}
}
```

---

## Authors

**Sarah Elyane Rashiwa**  
School of Electrical Engineering and Computer Science  
University of Ottawa  
Ottawa, Ontario, Canada  

**Paula Branco**  
School of Electrical Engineering and Computer Science  
University of Ottawa  
Ottawa, Ontario, Canada  

---

## License

This repository is released for research and reproducibility purposes.

A formal license will be added soon.

---

## Contact

For questions, please contact:

**Sarah Elyane Rashiwa**  
University of Ottawa  
Email: srash007@uottawa.ca


