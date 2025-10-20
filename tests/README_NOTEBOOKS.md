# Manual Testing Notebooks

This directory contains Jupyter notebooks for manual verification and testing of all plotting methods in `ddstartup/postprocessing`.

## 📚 Documentation

**Complete documentation has been moved to the Sphinx docs:**

👉 **See: `docs/user_guide/manual_testing/`**

### Quick Links:

- **Main Guide:** [`docs/user_guide/manual_testing/index.rst`](../docs/user_guide/manual_testing/index.rst)
- **Plot Overview:** [`docs/user_guide/manual_testing/plotting_overview.rst`](../docs/user_guide/manual_testing/plotting_overview.rst)
- **Testing Guide:** [`docs/user_guide/manual_testing/testing_guide.rst`](../docs/user_guide/manual_testing/testing_guide.rst)
- **Quick Reference:** [`docs/user_guide/manual_testing/quick_reference.rst`](../docs/user_guide/manual_testing/quick_reference.rst)
- **Notebooks Index:** [`docs/user_guide/manual_testing/notebooks_index.rst`](../docs/user_guide/manual_testing/notebooks_index.rst)

### Build & View Documentation:

```bash
cd docs/
make html
firefox _build/html/user_guide/manual_testing/index.html
```

## 📓 Notebooks in This Directory

### Postprocessing Plots
- `manual_contour_plots_verification.ipynb` - Contour and heatmap plots
- `manual_importance_matrix_verification.ipynb` - Effect size analysis
- `manual_shap_plots_verification.ipynb` - SHAP-style feature importance
- `manual_kmeans_plots_verification.ipynb` - K-means clustering
- `manual_kde_plots_verification.ipynb` - Kernel density estimation
- `manual_parcoords_plots_verification.ipynb` - Parallel coordinates
- `manual_pdf_plots_verification.ipynb` - PDF plots

### Physics Models
- `manual_lump_verification.ipynb` - Lump model verification
- `manual_tseeded_verification.ipynb` - T-seeded model verification

## 🚀 Quick Start

1. **View the documentation first** (see above)
2. **Open a notebook:**
   ```bash
   jupyter notebook manual_contour_plots_verification.ipynb
   ```
3. **Run all cells** to see examples
4. **Customize** for your data

## 📁 Output Files

Notebooks save test outputs to:
```
outputs/
├── manual_test_contour/
├── manual_test_importance/
├── manual_test_shap/
├── manual_test_kmeans/
└── ...
```

---

**For complete documentation, examples, and guides:** See `docs/user_guide/manual_testing/`
